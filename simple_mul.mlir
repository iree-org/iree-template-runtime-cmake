// This IREE compiler input represents a model as sourced from some frontend
// framework (JAX, PyTorch, TFLite, etc).
//
// Refer to the documentation for frontend-specific instructions:
//   https://openxla.github.io/iree/guides/ml-frameworks/

// Simple elementwise multiply:
//   %result = %lhs * %rhs
func.func @simple_mul(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = arith.mulf %lhs, %rhs : tensor<4xf32>
  return %result : tensor<4xf32>
}
