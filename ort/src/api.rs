use core::slice;
use std::{
    ffi::{c_char, c_void, CString},
    ptr::null,
    ptr::null_mut,
};

use ort_sys::{
    ONNXTensorElementDataType, ONNXType, OrtAllocator, OrtAllocatorType_OrtArenaAllocator, OrtApi,
    OrtEnv, OrtGetApiBase, OrtLoggingLevel, OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
    OrtMemType_OrtMemTypeDefault, OrtMemoryInfo, OrtModelMetadata, OrtRunOptions, OrtSession,
    OrtSessionOptions, OrtStatusPtr, OrtTensorTypeAndShapeInfo, OrtTypeInfo, OrtValue,
};

use crate::{ErrorStatus, TensorDataType, Wrapper, API_VERSION};

pub struct Api {
    api: &'static OrtApi,
}

#[allow(dead_code)]
impl Default for Api {
    fn default() -> Self {
        Self::new()
    }
}

impl Api {
    pub fn new() -> Self {
        unsafe {
            let base = OrtGetApiBase();
            let api = base.as_ref().unwrap().GetApi.unwrap()(API_VERSION);
            Api {
                api: api.as_ref().unwrap(),
            }
        }
    }

    pub fn create_env(&self, log_id: &str) -> Result<Wrapper<OrtEnv>, ErrorStatus> {
        unsafe {
            // FIXME: Don't leak!
            let log_id = CString::new(log_id).unwrap();
            let mut env = null_mut();

            self.api.CreateEnv.unwrap()(
                OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
                log_id.as_ptr(),
                &mut env,
            )
            .into_result(self.api)?;

            Ok(Wrapper {
                ptr: env,
                destructor: self.api.ReleaseEnv.unwrap(),
            })
        }
    }

    pub fn create_session_option(&self) -> Result<Wrapper<OrtSessionOptions>, ErrorStatus> {
        let mut opts = null_mut();
        unsafe { self.api.CreateSessionOptions.unwrap()(&mut opts).into_result(self.api)? }
        Ok(Wrapper {
            ptr: opts,
            destructor: self.api.ReleaseSessionOptions.unwrap(),
        })
    }

    pub fn create_session_from_file(
        &self,
        path: &str,
        env: *const OrtEnv,
        options: *const OrtSessionOptions,
    ) -> Result<Wrapper<OrtSession>, ErrorStatus> {
        let model_path = CString::new(path).unwrap().into_raw();
        let mut sess = null_mut();

        unsafe {
            self.api.CreateSession.unwrap()(env, model_path as *const _, options, &mut sess)
                .into_result(self.api)?
        };

        Ok(Wrapper {
            ptr: sess,
            destructor: self.api.ReleaseSession.unwrap(),
        })
    }

    pub fn create_session_from_bytes(
        &self,
        bytes: Vec<u8>,
        env: *const OrtEnv,
        opts: *const OrtSessionOptions,
    ) -> Result<Wrapper<OrtSession>, ErrorStatus> {
        let mut sess = null_mut();

        let n_bytes = bytes.len();
        unsafe {
            self.api.CreateSessionFromArray.unwrap()(
                env,
                bytes.as_ptr() as *const _,
                n_bytes,
                opts,
                &mut sess,
            )
            .into_result(self.api)?
        };
        Ok(Wrapper {
            ptr: sess,
            destructor: self.api.ReleaseSession.unwrap(),
        })
    }

    pub fn create_run_options(&self) -> Result<Wrapper<OrtRunOptions>, ErrorStatus> {
        let mut opts = null_mut();
        unsafe { self.api.CreateRunOptions.unwrap()(&mut opts).into_result(self.api)? }

        Ok(Wrapper {
            ptr: opts,
            destructor: self.api.ReleaseRunOptions.unwrap(),
        })
    }

    fn create_cpu_memory_info(&self) -> Result<Wrapper<OrtMemoryInfo>, ErrorStatus> {
        let alloc_ty = OrtAllocatorType_OrtArenaAllocator;
        let mem_ty = OrtMemType_OrtMemTypeDefault;

        let mut out = null_mut();
        unsafe {
            self.api.CreateCpuMemoryInfo.unwrap()(alloc_ty, mem_ty, &mut out)
                .into_result(self.api)?
        }
        Ok(Wrapper {
            ptr: out,
            destructor: self.api.ReleaseMemoryInfo.unwrap(),
        })
    }

    pub fn get_input_names(
        &self,
        sess: *const OrtSession,
    ) -> Result<Vec<*const c_char>, ErrorStatus> {
        let n = self.get_input_count(sess)?;
        (0..n).map(|i| self.get_input_name(sess, i)).collect()
    }

    fn get_input_count(&self, sess: *const OrtSession) -> Result<usize, ErrorStatus> {
        let mut n = 0;
        unsafe {
            self.api.SessionGetInputCount.unwrap()(sess, &mut n).into_result(self.api)?;
        }
        Ok(n)
    }

    fn get_input_name(
        &self,
        sess: *const OrtSession,
        idx: usize,
    ) -> Result<*const c_char, ErrorStatus> {
        let alloc = self.get_allocator()?;
        let mut out = null_mut();
        unsafe {
            self.api.SessionGetInputName.unwrap()(sess, idx, alloc, &mut out)
                .into_result(self.api)?;
        }
        Ok(out as *const _)
    }

    pub fn get_output_names(
        &self,
        sess: *const OrtSession,
    ) -> Result<Vec<*const c_char>, ErrorStatus> {
        let n = self.get_output_count(sess)?;
        (0..n).map(|i| self.get_output_name(sess, i)).collect()
    }

    fn get_output_count(&self, sess: *const OrtSession) -> Result<usize, ErrorStatus> {
        let mut n = 0;
        unsafe {
            self.api.SessionGetOutputCount.unwrap()(sess, &mut n).into_result(self.api)?;
        }
        Ok(n)
    }

    fn get_output_name(
        &self,
        sess: *const OrtSession,
        idx: usize,
    ) -> Result<*const c_char, ErrorStatus> {
        let alloc = self.get_allocator()?;
        let mut out = null_mut();
        unsafe {
            self.api.SessionGetOutputName.unwrap()(sess, idx, alloc, &mut out)
                .into_result(self.api)?;
        }
        Ok(out as *const _)
    }

    pub fn get_input_type_info(
        &self,
        sess: *const OrtSession,
        idx: usize,
    ) -> Result<Wrapper<OrtTypeInfo>, ErrorStatus> {
        let mut ptr = null_mut();
        unsafe {
            self.api.SessionGetInputTypeInfo.unwrap()(sess, idx, &mut ptr).into_result(self.api)?;
        }
        Ok(Wrapper {
            ptr,
            destructor: self.api.ReleaseTypeInfo.unwrap(),
        })
    }

    pub fn get_output_type_info(
        &self,
        sess: *const OrtSession,
        idx: usize,
    ) -> Result<Wrapper<OrtTypeInfo>, ErrorStatus> {
        let mut ptr = null_mut();
        unsafe {
            self.api.SessionGetOutputTypeInfo.unwrap()(sess, idx, &mut ptr)
                .into_result(self.api)?;
        }
        Ok(Wrapper {
            ptr,
            destructor: self.api.ReleaseTypeInfo.unwrap(),
        })
    }

    /// Get always the same pointer to the default allocator.
    pub fn get_allocator(&self) -> Result<*mut OrtAllocator, ErrorStatus> {
        let mut out = null_mut();
        unsafe {
            self.api.GetAllocatorWithDefaultOptions.unwrap()(&mut out).into_result(self.api)?;
        }
        Ok(out)
    }

    pub fn run(
        &self,
        sess: *mut OrtSession,
        run_options: *const OrtRunOptions,
        in_names: &[*const i8],
        in_values: &[*const OrtValue],
        out_names: &[*const i8],
    ) -> Result<Vec<Wrapper<OrtValue>>, ErrorStatus> {
        let mut out_values = vec![null_mut(); out_names.len()];
        unsafe {
            self.api.Run.unwrap()(
                sess,
                run_options,
                in_names.as_ptr(),
                in_values.as_ptr(),
                in_names.len(),
                out_names.as_ptr(),
                out_names.len(),
                out_values.as_mut_ptr(), // TODO: Why is this `*mut *mut`?!
            )
            .into_result(self.api)?;
        }
        let wrapped = out_values.into_iter().map(|ptr| Wrapper {
            ptr,
            destructor: self.api.ReleaseValue.unwrap(),
        });
        Ok(wrapped.collect())
    }

    /// Create a tensor backed by a user provided buffer. The returned
    /// `OrtValue` is only valid for the lifetime of `data`.
    pub unsafe fn create_tensor_with_borrowed_data<T>(
        &self,
        data: &mut [T],
        shape: &[usize],
    ) -> Result<Wrapper<OrtValue>, ErrorStatus>
    where
        T: TensorDataType + Sized,
    {
        let mem_info = self.create_cpu_memory_info()?;
        let data_n_bytes = std::mem::size_of_val(data);
        let shape_len = shape.len();
        let ty = T::tensor_dtype();
        let mut out = null_mut();

        unsafe {
            self.api.CreateTensorWithDataAsOrtValue.unwrap()(
                mem_info.ptr,
                data.as_mut_ptr() as *mut _,
                data_n_bytes,
                shape.as_ptr() as _,
                shape_len,
                ty,
                &mut out,
            )
            .into_result(self.api)?
        }
        Ok(Wrapper {
            ptr: out,
            destructor: self.api.ReleaseValue.unwrap(),
        })
    }

    /// Create a tensor that owns a copy of the `data`.
    pub fn create_tensor_with_copied_data<T>(
        &self,
        data: &[T],
        shape: &[usize],
    ) -> Result<Wrapper<OrtValue>, ErrorStatus>
    where
        T: TensorDataType + Sized + Copy,
    {
        let alloc = self.get_allocator()?;

        let shape_len = shape.len();
        let ty = T::tensor_dtype();

        let mut ort_value = null_mut();

        unsafe {
            self.api.CreateTensorAsOrtValue.unwrap()(
                alloc,
                shape.as_ptr() as _,
                shape_len,
                ty,
                &mut ort_value,
            )
            .into_result(self.api)?
        };

        let ort_buffer =
            unsafe { slice::from_raw_parts_mut(self.get_tensor_data_mut(ort_value)?, data.len()) };
        ort_buffer.clone_from_slice(data);
        Ok(Wrapper {
            ptr: ort_value,
            destructor: self.api.ReleaseValue.unwrap(),
        })
    }

    /// Create a string tensor by copying the data into the buffer of the returned `OrtValue`.
    pub fn create_string_tensor(
        &self,
        data: &[&str],
        shape: &[usize],
    ) -> Result<Wrapper<OrtValue>, ErrorStatus> {
        let alloc = self.get_allocator()?;

        let shape_len = shape.len();
        let ty = String::tensor_dtype();

        let mut ort_value = null_mut();

        unsafe {
            self.api.CreateTensorAsOrtValue.unwrap()(
                alloc,
                shape.as_ptr() as _,
                shape_len,
                ty,
                &mut ort_value,
            )
            .into_result(self.api)?
        };

        // Null terminated Cstrings
        let cstrings: Vec<_> = data
            .iter()
            .map(|&s| CString::new(s).expect("String contains null bytes"))
            .collect();

        let cstrs: Vec<_> = cstrings.iter().map(|s| s.as_c_str().as_ptr()).collect();

        unsafe { self.api.FillStringTensor.unwrap()(ort_value, cstrs.as_ptr(), data.len()) };

        Ok(Wrapper {
            ptr: ort_value,
            destructor: self.api.ReleaseValue.unwrap(),
        })
    }

    pub unsafe fn get_tensor_data_mut<T>(
        &self,
        value: *mut OrtValue,
    ) -> Result<*mut T, ErrorStatus> {
        let mut out = null_mut();
        unsafe {
            self.api.GetTensorMutableData.unwrap()(value, &mut out).into_result(self.api)?;
            Ok(out as *mut _)
        }
    }

    pub fn get_string_tensor_data_length(
        &self,
        value: *const OrtValue,
    ) -> Result<usize, ErrorStatus> {
        let mut len = 0;
        unsafe {
            self.api.GetStringTensorDataLength.unwrap()(value, &mut len).into_result(self.api)?
        };
        Ok(len)
    }

    /// Get a contiguous copy of a string tensor buffer with the respective `offsets`.
    pub unsafe fn get_string_tensor_buffer(
        &self,
        value: *const OrtValue,
        offset_len: usize,
    ) -> Result<(Vec<u8>, Vec<usize>), ErrorStatus> {
        let buf_len = self.get_string_tensor_data_length(value)?;

        let mut buf: Vec<u8> = vec![0; buf_len];

        let mut offsets = vec![0; offset_len];
        self.api.GetStringTensorContent.unwrap()(
            value,
            buf.as_mut_ptr() as *mut _,
            buf_len,
            offsets.as_mut_ptr(),
            offset_len,
        );
        Ok((buf, offsets))
    }

    /// Free the pointed-to memory using the provided allocator
    pub unsafe fn free(
        &self,
        alloc: *mut OrtAllocator,
        ptr: *mut c_void,
    ) -> Result<(), ErrorStatus> {
        self.api.AllocatorFree.unwrap()(alloc, ptr).into_result(self.api)
    }

    pub unsafe fn get_type_info_from_ort_value(
        &self,
        value: *const OrtValue,
    ) -> Result<Wrapper<OrtTypeInfo>, ErrorStatus> {
        let mut ptr = null_mut();
        self.api.GetTypeInfo.unwrap()(value, &mut ptr).into_result(self.api)?;
        Ok(Wrapper {
            ptr,
            destructor: self.api.ReleaseTypeInfo.unwrap(),
        })
    }

    pub unsafe fn get_onnx_type(&self, info: *const OrtTypeInfo) -> Result<ONNXType, ErrorStatus> {
        let mut out = 0;
        self.api.GetOnnxTypeFromTypeInfo.unwrap()(info, &mut out).into_result(self.api)?;
        Ok(out)
    }

    /// Get the corresponding `OrtTensorTypeAndShapeInfo` object from
    /// a `OrtTypeInfo`.  The `OrtTensorTypeAndShapeInfo` is freed
    /// when the `OrtTypeInfo` object is freed. Returns a null pointer
    /// if `OrtTypeInfo` does not describe a tensor.
    pub unsafe fn type_info_as_tensor_type_info<'info>(
        &self,
        type_info: &'info OrtTypeInfo,
    ) -> Result<&'info OrtTensorTypeAndShapeInfo, ErrorStatus> {
        let mut ptr = null();
        self.api.CastTypeInfoToTensorInfo.unwrap()(type_info, &mut ptr).into_result(self.api)?;
        Ok(ptr.as_ref().unwrap())
    }

    pub unsafe fn get_tensor_data_type(
        &self,
        tensor_info: *const OrtTensorTypeAndShapeInfo,
    ) -> Result<ONNXTensorElementDataType, ErrorStatus> {
        let mut out = 0;
        self.api.GetTensorElementType.unwrap()(tensor_info, &mut out).into_result(self.api)?;
        Ok(out)
    }

    unsafe fn get_tensor_rank(
        &self,
        tensor_info: *const OrtTensorTypeAndShapeInfo,
    ) -> Result<usize, ErrorStatus> {
        let mut out = 0;
        self.api.GetDimensionsCount.unwrap()(tensor_info, &mut out).into_result(self.api)?;
        Ok(out)
    }

    /// Get tensor shape.
    ///
    /// Dimensions with missing numerical length information (e.g. for session
    /// inputs/outputs) are denoted by `-1`.
    pub unsafe fn get_tensor_shape(
        &self,
        tensor_info: *const OrtTensorTypeAndShapeInfo,
    ) -> Result<Vec<i64>, ErrorStatus> {
        let rank = self.get_tensor_rank(tensor_info)?;
        let mut out: Vec<i64> = vec![0; rank];

        self.api.GetDimensions.unwrap()(tensor_info, out.as_mut_ptr() as *mut _, rank)
            .into_result(self.api)?;
        Ok(out)
    }

    pub unsafe fn get_tensor_shape_symbolic(
        &self,
        tensor_info: *const OrtTensorTypeAndShapeInfo,
    ) -> Result<Vec<*const i8>, ErrorStatus> {
        let rank = self.get_tensor_rank(tensor_info)?;
        let mut out: Vec<*const i8> = vec![null(); rank];

        self.api.GetSymbolicDimensions.unwrap()(tensor_info, out.as_mut_ptr() as *mut _, rank)
            .into_result(self.api)?;
        Ok(out)
    }

    pub unsafe fn get_tensor_element_count(
        &self,
        tensor_info: *const OrtTensorTypeAndShapeInfo,
    ) -> Result<usize, ErrorStatus> {
        let mut n = 0;
        self.api.GetTensorShapeElementCount.unwrap()(tensor_info, &mut n).into_result(self.api)?;

        Ok(n)
    }

    pub unsafe fn set_log_severity(
        &self,
        session_options: *mut OrtSessionOptions,
        level: OrtLoggingLevel,
    ) -> Result<(), ErrorStatus> {
        self.api.SetSessionLogSeverityLevel.unwrap()(session_options, level as _)
            .into_result(self.api)
    }

    #[allow(clippy::type_complexity)]
    pub fn get_model_metadata_map(
        &self,
        sess: *const OrtSession,
    ) -> Result<Vec<(Wrapper<i8>, Wrapper<i8>)>, ErrorStatus> {
        unsafe {
            let meta = self.get_model_metadata(sess)?;
            let keys = self.get_model_metadata_keys(meta.ptr)?;

            let mut out = Vec::with_capacity(keys.len());
            for key in keys.into_iter() {
                let value = self.get_model_metadata_value(meta.ptr, key.ptr)?;

                out.push((key, value));
            }
            Ok(out)
        }
    }

    unsafe fn get_model_metadata(
        &self,
        sess: *const OrtSession,
    ) -> Result<Wrapper<OrtModelMetadata>, ErrorStatus> {
        let mut ptr = null_mut();
        self.api.SessionGetModelMetadata.unwrap()(sess, &mut ptr).into_result(self.api)?;

        Ok(Wrapper {
            ptr,
            destructor: self.api.ReleaseModelMetadata.unwrap(),
        })
    }

    unsafe fn get_model_metadata_keys(
        &self,
        meta: *const OrtModelMetadata,
    ) -> Result<Vec<Wrapper<i8>>, ErrorStatus> {
        let mut n = 0;
        let mut ptrs = null_mut();
        let alloc = self.get_allocator()?;
        self.api.ModelMetadataGetCustomMetadataMapKeys.unwrap()(meta, alloc, &mut ptrs, &mut n)
            .into_result(self.api)?;

        let out: &mut [*mut i8] = slice::from_raw_parts_mut(ptrs, n as usize);
        Ok(out
            .iter_mut()
            .map(|&mut ptr| Wrapper {
                ptr,
                destructor: dealloc_chars,
            })
            .collect())
    }

    unsafe fn get_model_metadata_value(
        &self,
        meta: *const OrtModelMetadata,
        key: *const i8,
    ) -> Result<Wrapper<i8>, ErrorStatus> {
        let mut ptr = null_mut();
        let alloc = self.get_allocator()?;
        self.api.ModelMetadataLookupCustomMetadataMap.unwrap()(meta, alloc, key, &mut ptr)
            .into_result(self.api)?;
        Ok(Wrapper {
            ptr,
            destructor: dealloc_chars,
        })
    }
}

unsafe extern "C" fn dealloc_chars(ptr: *mut i8) {
    let api = Api::new();
    let alloc = api.get_allocator().unwrap();
    api.free(alloc, ptr as *mut _).unwrap();
}

trait IntoResult {
    fn into_result(self, api: &OrtApi) -> Result<(), ErrorStatus>;
}

impl IntoResult for OrtStatusPtr {
    fn into_result(self, api: &OrtApi) -> Result<(), ErrorStatus> {
        if self.is_null() {
            Ok(())
        } else {
            Err(ErrorStatus::new(self, api))
        }
    }
}
