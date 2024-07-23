# Ascend NPU docs:
# https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/aclpythondevg/aclpythondevg_0003.html
# https://github.com/Ascend/samples/tree/master/inference/acllite

import atexit
import numpy as np
import typing
import acl

ACL_HOST = 1
ACL_ERROR_NONE = 0
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_FLOAT = 0
ACL_INT32 = 3
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_ERROR_REPEAT_INITIALIZE = 100002

ACL_TO_NUMPY = {  # mapping from ACL to Numpy types
    ACL_FLOAT: np.float32,
    ACL_INT32: np.int32,
    ACL_UINT32: np.uint32,
    ACL_INT64: np.int64,
    ACL_UINT64: np.uint64,
}

class AclModelMetadata(typing.NamedTuple):
    name: str
    shape: typing.Tuple[int]
    dtype: typing.Any


def _check_ret(fn_name, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception(f"Call to {fn_name} failed: {ret}")


class AclRuntime:
    def __init__(self, model_path='model_path', device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.run_mode = None
        self.model_desc = None
        
        self.input_len = 0
        self.input_dataset = None
        self.input_buffer_ptr = []
        self.input_info: typing.List[AclModelMetadata] = []
        
        self.output_dataset = None
        self.output_info: typing.List[AclModelMetadata] = None
        
        self.host_buffer = None
        self.host_buffer_size = 0
        
        self._init_resource()
        self._init_model()
        self._init_input()
        
        self._init_output()

    def _init_resource(self):
        ret = acl.init()
        if ret != ACL_ERROR_REPEAT_INITIALIZE:  # already initialized, ignore
            _check_ret("acl.init", ret)
        
        ret = acl.rt.set_device(self.device_id)
        _check_ret("acl.rt.set_device", ret)
        
        self.context, ret = acl.rt.create_context(self.device_id)
        _check_ret("acl.rt.create_context", ret)
        
        self.run_mode, ret = acl.rt.get_run_mode()
        _check_ret("acl.rt.get_run_mode", ret)

    def _init_model(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        _check_ret("acl.mdl.load_from_file", ret)
        
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        _check_ret("acl.mdl.get_desc", ret)

    def _init_input(self):
        self.input_len = acl.mdl.get_num_inputs(self.model_desc)
        
        self.input_dataset = acl.mdl.create_dataset()        
        for idx in range(self.input_len):
            # memory allocation
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, idx)
            buffer_ptr, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            _check_ret("acl.rt.malloc", ret)
            
            data_buffer = acl.create_data_buffer(buffer_ptr, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
            _check_ret("acl.mdl.add_dataset_buffer", ret)
            
            self.input_buffer_ptr.append(buffer_ptr)
            
            # metadata retrieval
            dims, ret = acl.mdl.get_input_dims(self.model_desc, idx)
            _check_ret("acl.mdl.get_input_dims", ret)
            name = acl.mdl.get_input_name_by_index(self.model_desc, idx)
            datatype = acl.mdl.get_input_data_type(self.model_desc, idx)
            self.input_info.append(AclModelMetadata(name, tuple(dims["dims"]), ACL_TO_NUMPY.get(datatype)))
        
    def _init_output(self):        
        idx = 0  # only 1 output is supported
        self.output_dataset = acl.mdl.create_dataset()
        
        temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, idx)
        temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        dataset_buffer = acl.create_data_buffer(temp_buffer, temp_buffer_size)
        _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, dataset_buffer)
        _check_ret("acl.mdl.add_dataset_buffer", ret)
            
        dims, ret = acl.mdl.get_output_dims(self.model_desc, idx)
        _check_ret("acl.mdl.get_output_dims", ret)
        name = acl.mdl.get_input_name_by_index(self.model_desc, idx)
        datatype = acl.mdl.get_output_data_type(self.model_desc, idx)
        self.output_info = AclModelMetadata(name, tuple(dims["dims"]), ACL_TO_NUMPY.get(datatype))
    
        # allocate memory at the host
        self.host_buffer_size = temp_buffer_size
        self.host_buffer, ret = acl.rt.malloc_host(self.host_buffer_size)
        _check_ret("acl.rt.malloc_host", ret)

    def __str__(self):
        return f"{type(self)} input: {self.input_info}, output: {self.output_info}"
    
    def __del__(self):            
        if self.input_dataset:
            self._release_dataset(self.input_dataset)
            self.input_dataset = None

        if self.output_dataset:
            self._release_dataset(self.output_dataset)
            self.output_dataset = None
            
        if self.host_buffer:
            ret = acl.rt.free_host(self.host_buffer)
            _check_ret("acl.rt.free_host", ret)
            self.host_buffer = None
            
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            _check_ret("acl.mdl.unload", ret)
            self.model_id = None

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            _check_ret("acl.mdl.destroy_desc", ret)
            self.model_desc = None
            
        if self.context:
            ret = acl.rt.destroy_context(self.context)
            _check_ret("acl.rt.destroy_context", ret)
            self.context = None
            
    def _release_dataset(self, dataset):
        num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf:
                self._release_databuffer(data_buf)

        ret = acl.mdl.destroy_dataset(dataset)
        _check_ret("acl.mdl.destroy_dataset", ret)

    def _release_databuffer(self, data_buffer):
        data_addr = acl.get_data_buffer_addr(data_buffer)
        if data_addr:
            acl.rt.free(data_addr)

        ret = acl.destroy_data_buffer(data_buffer)
        _check_ret("acl.destroy_data_buffer", ret)

    def run(self, *input_data):
        ret = acl.rt.set_context(self.context)
        _check_ret("acl.rt.set_context", ret)
        self._copy_input(*input_data)

        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        _check_ret("acl.mdl.execute", ret)
        
        result = self._output_dataset_to_numpy()
        return result

    def _copy_input(self, *input_data):
        if len(input_data) != len(self.input_buffer_ptr):
            raise ValueError(f"The model expects {len(self.input_buffer_ptr)} inputs, but {len(input_data)} was provided")
        
        for i, data in enumerate(input_data):
            if data.shape != self.input_info[i].shape:
                raise ValueError(f"Shape of input with index {i} is expected to be {self.input_info[i].shape}, but {data.shape} was provided.")
            if data.dtype != self.input_info[i].dtype:
                raise ValueError(f"Type of input with index {i} is expected to be {self.input_info[i].dtype}, but {data.dtype} was provided.")
            
            input_buffer = acl.util.bytes_to_ptr(data.tobytes())
            ret = acl.rt.memcpy(self.input_buffer_ptr[i], data.nbytes, input_buffer, data.nbytes, ACL_MEMCPY_HOST_TO_DEVICE)
            _check_ret("acl.rt.memcpy", ret)
        
    def _output_dataset_to_numpy(self):
        idx = 0  # only 1 output is supported
        buffer = acl.mdl.get_dataset_buffer(self.output_dataset, idx)
        data = acl.get_data_buffer_addr(buffer)
        size = acl.get_data_buffer_size(buffer)
        
        # if self.run_mode == ACL_HOST:
        ret = acl.rt.memcpy(self.host_buffer, self.host_buffer_size, data, size, ACL_MEMCPY_DEVICE_TO_HOST)
        _check_ret("acl.rt.memcpy", ret)

        # convert to numpy array
        byte_buffer = acl.util.ptr_to_bytes(self.host_buffer, self.host_buffer_size)
        shape = self.output_info.shape
        dtype = self.output_info.dtype
        data_array = np.frombuffer(byte_buffer, dtype=dtype).reshape(shape)
        return data_array


def _acl_finalize():
    ret = acl.finalize()        
    _check_ret("acl.finalize", ret)
        
atexit.register(_acl_finalize)
