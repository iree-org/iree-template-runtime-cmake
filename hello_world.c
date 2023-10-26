// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

// When using the iree_runtime_* APIs this is the only include required:
#include <iree/runtime/api.h>

static iree_status_t iree_runtime_demo_perform_mul(
    iree_runtime_session_t* session);

//===----------------------------------------------------------------------===//
// Entry point and session management
//===----------------------------------------------------------------------===//

// Takes the device to use and module to load on the command line.
// This would live in your application startup/shutdown code or scoped to the
// usage of IREE. Creating and destroying instances may be expensive and should
// be avoided.
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: hello_world device module.vmfb\n");
    return 1;
  }
  const char* device_uri = argv[1];
  const char* module_path = argv[2];

  // Setup the shared runtime instance.
  // An application should usually only have one of these and share it across
  // all of the sessions it has. The instance is thread-safe while the
  // sessions are only thread-compatible (you need to lock around them if
  // multiple threads will be using them). Asynchronous execution allows for
  // a single thread (or short-duration lock) to use the session for launching
  // invocations while allowing for the invocations to overlap in execution.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance);

  // Create the HAL device used to run the workloads. This should be shared
  // across multiple sessions unless isolation is required (rare outside of
  // multi-tenant servers). The device may own limited or expensive resources
  // (like thread pools) and should be persisted for as long as possible.
  //
  // This form of iree_hal_create_device allows the user to pick the device on
  // the command line out of any available devices with their HAL drivers
  // compiled into the runtime. iree_runtime_instance_try_create_default_device
  // and other APIs are available to create the default device and
  // `iree-run-module --dump_devices` and other tools can be used to show the
  // available devices. Integrators can also enumerate HAL drivers and devices
  // if they want to present options to the end user.
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(
        iree_runtime_instance_driver_registry(instance),
        iree_make_cstring_view(device_uri),
        iree_runtime_instance_host_allocator(instance), &device);
  }

  // Set up the session to run the demo module.
  // Sessions are like OS processes and are used to isolate module state such as
  // the variables used within the module. The same module loaded into two
  // sessions will see their own private state.
  //
  // A real application would load its modules (at startup, on-demand, etc) and
  // retain them somewhere to be reused. Startup time and likelihood of failure
  // varies across different HAL backends; the synchronous CPU backend is nearly
  // instantaneous and will never fail (unless out of memory) while the Vulkan
  // backend may take significantly longer and fail if there are unsupported
  // or unavailable devices.
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  // Load the compiled user module from a file.
  // Applications could specify files, embed the outputs directly in their
  // binaries, fetch them over the network, etc. Modules are linked in the order
  // they are added and custom modules usually come before compiled modules.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(session,
                                                                   module_path);
  }

  // Build and issue the call - here just one we do for this sample but in a
  // real application the session should be reused as much as possible. Always
  // keep state within the compiled module instead of externalizing and passing
  // it as arguments/results as IREE cannot optimize external state.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_demo_perform_mul(session);
  }

  // Release the session and free all cached resources.
  iree_runtime_session_release(session);

  // Release shared device once all sessions using it have been released.
  iree_hal_device_release(device);

  // Release the shared instance - it will be deallocated when all sessions
  // using it have been released (here it is deallocated immediately).
  iree_runtime_instance_release(instance);

  int ret = (int)iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    // Dump nice status messages to stderr on failure.
    // An application can route these through its own logging infrastructure as
    // needed. Note that the status is a handle and must be freed!
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// Call a function within a module with buffer views
//===----------------------------------------------------------------------===//
// The inputs and outputs of a call are reusable across calls (and possibly
// across sessions depending on device compatibility) and can be setup by the
// application as needed. For example, an application could perform
// multi-threaded buffer view creation and then issue the call from a single
// thread when all inputs are ready. This simple demo just allocates them
// per-call and throws them away.

// Sets up and calls the simple_mul function and dumps the results:
// func.func @simple_mul(
//     %lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32>
//
// NOTE: this is a demo and as such this performs no memoization; a real
// application could reuse a lot of these structures and cache lookups of
// iree_vm_function_t to reduce the amount of per-call overhead.
static iree_status_t iree_runtime_demo_perform_mul(
    iree_runtime_session_t* session) {
  // Initialize the call to the function.
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.simple_mul"), &call));

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    // %lhs: tensor<4xf32>
    iree_hal_buffer_view_t* lhs = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t lhs_shape[1] = {4};
      static const float lhs_data[4] = {1.0f, 1.1f, 1.2f, 1.3f};
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator,
          // Shape rank and dimensions:
          IREE_ARRAYSIZE(lhs_shape), lhs_shape,
          // Element type:
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          // Encoding type:
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              // Where to allocate (host or device):
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              // Access to allow to this memory (this is .rodata so READ only):
              .access = IREE_HAL_MEMORY_ACCESS_READ,
              // Intended usage of the buffer (transfers, dispatches, etc):
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          // The actual heap buffer to wrap or clone and its allocator:
          iree_make_const_byte_span(lhs_data, sizeof(lhs_data)),
          // Buffer view + storage are returned and owned by the caller:
          &lhs);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, lhs, /*max_element_count=*/4096, host_allocator));
      // Add to the call inputs list (which retains the buffer view).
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, lhs);
    }
    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(lhs);

    fprintf(stdout, "\n * \n");

    // %rhs: tensor<4xf32>
    iree_hal_buffer_view_t* rhs = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t rhs_shape[1] = {4};
      static const float rhs_data[4] = {10.0f, 100.0f, 1000.0f, 10000.0f};
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator, IREE_ARRAYSIZE(rhs_shape), rhs_shape,
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              .access = IREE_HAL_MEMORY_ACCESS_READ,
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          iree_make_const_byte_span(rhs_data, sizeof(rhs_data)), &rhs);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, rhs, /*max_element_count=*/4096, host_allocator));
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, rhs);
    }
    iree_hal_buffer_view_release(rhs);
  }

  // Synchronously perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  fprintf(stdout, "\n = \n");

  // Dump the function outputs.
  iree_hal_buffer_view_t* result = NULL;
  if (iree_status_is_ok(status)) {
    // Try to get the first call result as a buffer view.
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }
  if (iree_status_is_ok(status)) {
    // This prints the buffer view out but an application could read its
    // contents, pass it to another call, etc.
    status = iree_hal_buffer_view_fprint(
        stdout, result, /*max_element_count=*/4096, host_allocator);
  }
  iree_hal_buffer_view_release(result);

  iree_runtime_call_deinitialize(&call);
  return status;
}
