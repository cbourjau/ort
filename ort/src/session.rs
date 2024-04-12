use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::path::Path;

use ort_sys::{OrtAllocator, OrtEnv, OrtRunOptions, OrtSession, OrtValue};

use crate::type_info::TypeInfo;
use crate::{Api, ErrorStatus, IntoValue, Value, Wrapper};

pub struct Session {
    api: Api,
    _env: Wrapper<OrtEnv>,
    ort_sess: Wrapper<OrtSession>,
    alloc: *mut OrtAllocator,
    input_names: Vec<*const c_char>,
    output_names: Vec<*const c_char>,
}

impl Session {
    pub fn from_bytes(model: Vec<u8>) -> Result<Self, ErrorStatus> {
        let api = Api::new();

        let alloc = api.get_allocator()?;
        let env = api.create_env("Runtime environment")?;
        let opts = api.create_session_option()?;

        let ort_sess = api.create_session_from_bytes(model, env.ptr, opts.ptr)?;

        let input_names = api.get_input_names(ort_sess.ptr)?;
        let output_names = api.get_output_names(ort_sess.ptr)?;

        Ok(Self {
            api,
            _env: env,
            alloc,
            input_names,
            output_names,
            ort_sess,
        })
    }

    pub fn from_path(model: &Path) -> Result<Self, ErrorStatus> {
        let api = Api::new();

        let alloc = api.get_allocator()?;
        let env = api.create_env("Runtime environment")?;
        let opts = api.create_session_option()?;

        let ort_sess = api.create_session_from_file(model.to_str().unwrap(), env.ptr, opts.ptr)?;

        let input_names = api.get_input_names(ort_sess.ptr)?;
        let output_names = api.get_output_names(ort_sess.ptr)?;

        Ok(Self {
            api,
            _env: env,
            alloc,
            input_names,
            output_names,
            ort_sess,
        })
    }

    pub fn run(
        &self,
        inputs: HashMap<&str, &Value>,
        run_options: Option<Wrapper<OrtRunOptions>>,
    ) -> Result<HashMap<&str, Value>, ErrorStatus> {
        let inputs = inputs
            .into_iter()
            .map(|(k, v)| Ok::<_, ErrorStatus>((k, v.ref_ort_value())))
            .collect::<Result<HashMap<_, _>, ErrorStatus>>()?;
        let run_options = match run_options {
            None => self.api.create_run_options()?,
            Some(run_options) => run_options,
        };
        let output = self.run_ort_values(&inputs, run_options)?;

        output
            .into_iter()
            .map(|(k, ort_v)| Ok((k, ort_v.into_value()?)))
            .collect()
    }

    pub fn get_input_infos(&self) -> Result<Vec<(&str, TypeInfo)>, ErrorStatus> {
        let mut out = Vec::new();
        for (idx, k) in self.input_names_iter().enumerate() {
            let info = self.api.get_input_type_info(self.ort_sess.ptr, idx)?;
            out.push((k, TypeInfo::new(&self.api, &info)?));
        }
        Ok(out)
    }
    pub fn get_output_infos(&self) -> Result<Vec<(&str, TypeInfo)>, ErrorStatus> {
        let mut out = Vec::new();
        for (idx, k) in self.output_names_iter().enumerate() {
            let info = self.api.get_output_type_info(self.ort_sess.ptr, idx)?;
            out.push((k, TypeInfo::new(&self.api, &info)?));
        }
        Ok(out)
    }

    pub fn get_model_metadata(&self) -> Result<HashMap<String, String>, ErrorStatus> {
        let mut out = HashMap::new();
        for (k, v) in self
            .api
            .get_model_metadata_map(self.ort_sess.ptr)?
            .into_iter()
        {
            let k = unsafe { CStr::from_ptr(k.ptr).to_str().unwrap() };
            let v = unsafe { CStr::from_ptr(v.ptr).to_str().unwrap() };
            out.insert(k.to_string(), v.to_string());
        }
        Ok(out)
    }

    pub(crate) fn run_ort_values<'a>(
        &'a self,
        inputs: &HashMap<&str, &Wrapper<OrtValue>>,
        run_options: Wrapper<OrtRunOptions>,
    ) -> Result<HashMap<&'a str, Wrapper<OrtValue>>, ErrorStatus> {
        let mut in_values = vec![];

        for k in self.input_names.iter() {
            let k = unsafe { CStr::from_ptr(*k as *const _).to_str().unwrap() };
            in_values.push(inputs.get(k).unwrap().ptr);
        }

        let out_values = self
            .api
            .run(
                self.ort_sess.ptr,
                run_options.ptr,
                self.input_names.as_slice(),
                in_values
                    .iter()
                    .map(|item| item.cast_const())
                    .collect::<Vec<_>>()
                    .as_slice(),
                self.output_names.as_slice(),
            )
            .unwrap();

        let mut out = HashMap::new();

        for (k, v) in self.output_names_iter().zip(out_values.into_iter()) {
            out.insert(k, v);
        }

        Ok(out)
    }

    fn input_names_iter(&self) -> impl Iterator<Item = &str> {
        self.input_names
            .iter()
            .map(|k| unsafe { CStr::from_ptr(*k as *const _).to_str().unwrap() })
    }

    fn output_names_iter(&self) -> impl Iterator<Item = &str> {
        self.output_names
            .iter()
            .map(|k| unsafe { CStr::from_ptr(*k as *const _).to_str().unwrap() })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for n in self.input_names.drain(..) {
            unsafe { self.api.free(self.alloc, n as *mut _) }.unwrap();
        }

        for n in self.output_names.drain(..) {
            unsafe { self.api.free(self.alloc, n as *mut _) }.unwrap();
        }
    }
}

unsafe impl Sync for Session {}
