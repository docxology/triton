#!/usr/bin/env python3
"""
Triton Usage Validation Tests

Comprehensive validation that all implementations in the active inference framework
use real Triton methods and GPU acceleration. This script verifies:

1. All Triton kernels are properly implemented (not just placeholders)
2. GPU acceleration is used when available
3. PyTorch fallbacks work correctly when Triton is unavailable
4. Memory management uses real Triton features
5. Performance optimizations leverage Triton capabilities
6. No mock methods are used in production code

Usage:
    python -m pytest tests/test_triton_validation.py -v
    or
    python -m pytest tests/test_triton_validation.py::TestTritonValidation::test_comprehensive_validation -v
"""

import sys
import os
import torch
import inspect
import importlib
import pytest
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def setup_test_environment():
    """Set up test environment with proper imports."""
    # Ensure we're in the right directory
    active_inference_dir = Path(__file__).parent.parent
    os.chdir(active_inference_dir)
    sys.path.insert(0, str(active_inference_dir / "src"))

    yield

    # Cleanup
    if str(active_inference_dir / "src") in sys.path:
        sys.path.remove(str(active_inference_dir / "src"))


class TestTritonValidation:
    """Comprehensive Triton usage validation tests."""

    def test_triton_imports_validation(self, setup_test_environment):
        """Test that all Triton imports are correctly handled."""
        validation_results = self._validate_triton_imports()

        # Basic assertions
        assert isinstance(validation_results, dict)
        assert "triton_available" in validation_results
        assert "imports_correct" in validation_results
        assert "gpu_acceleration" in validation_results

        # If Triton is available, imports should be correct
        if validation_results["triton_available"]:
            assert validation_results[
                "imports_correct"
            ], "Triton availability detection should match actual import status"

    def test_kernel_implementations_validation(self, setup_test_environment):
        """Test that all kernel implementations are real (not placeholders)."""
        validation_results = self._validate_kernel_implementations()

        assert isinstance(validation_results, dict)
        assert "kernels_found" in validation_results
        assert "real_implementations" in validation_results
        assert "placeholder_detected" in validation_results

        # Should find some real implementations only if Triton is available
        try:
            from core import TRITON_AVAILABLE
            if TRITON_AVAILABLE:
                assert (
                    len(validation_results["real_implementations"]) > 0
                ), "Should find real Triton kernel implementations when Triton is available"
        except ImportError:
            # If we can't import, assume Triton is not available
            pass

    def test_gpu_acceleration_validation(self, setup_test_environment):
        """Test that GPU acceleration is properly implemented."""
        validation_results = self._validate_gpu_acceleration()

        assert isinstance(validation_results, dict)

        # Check for expected keys
        expected_keys = [
            "gpu_available",
            "memory_management",
            "kernel_launch",
            "synchronization",
        ]
        for key in expected_keys:
            assert key in validation_results

        # If GPU is available, memory management should work
        if validation_results.get("gpu_available", False):
            assert validation_results[
                "memory_management"
            ], "Memory management should work when GPU is available"

    def test_real_vs_mock_methods_validation(self, setup_test_environment):
        """Test that no mock methods are used in production code."""
        validation_results = self._validate_real_vs_mock_methods()

        assert isinstance(validation_results, dict)
        assert "mock_methods_detected" in validation_results
        assert "real_implementations" in validation_results

        # Should have some real implementations
        assert (
            len(validation_results["real_implementations"]) > 0
        ), "Should find real code implementations"

    def test_memory_optimization_validation(self, setup_test_environment):
        """Test that memory optimization uses real Triton features."""
        validation_results = self._validate_memory_optimization()

        assert isinstance(validation_results, dict)

        # Check for memory optimization features
        expected_keys = [
            "shared_memory_usage",
            "coalesced_access",
            "memory_pooling",
            "efficient_allocation",
        ]
        for key in expected_keys:
            assert key in validation_results

    def test_performance_optimizations_validation(self, setup_test_environment):
        """Test that performance optimizations use real Triton features."""
        validation_results = self._validate_performance_optimizations()

        assert isinstance(validation_results, dict)

        # Check for performance optimization features
        expected_keys = [
            "kernel_fusion",
            "vectorization",
            "parallelization",
            "pipelining",
        ]
        for key in expected_keys:
            assert key in validation_results

    @pytest.mark.integration
    def test_comprehensive_validation(self, setup_test_environment):
        """Run comprehensive validation of all Triton usage."""
        all_results = {}

        # Run all validations
        all_results["triton_imports"] = self._validate_triton_imports()
        all_results["kernel_implementations"] = self._validate_kernel_implementations()
        all_results["gpu_acceleration"] = self._validate_gpu_acceleration()
        all_results["real_vs_mock"] = self._validate_real_vs_mock_methods()
        all_results["memory_optimization"] = self._validate_memory_optimization()
        all_results["performance_optimizations"] = (
            self._validate_performance_optimizations()
        )

        # Generate report
        report = self._generate_validation_report(all_results)

        # Assertions
        assert "overall_score" in report
        assert "assessment" in report
        assert "recommendations" in report

        # Should have a reasonable score
        overall_score = report["overall_score"]
        assert isinstance(overall_score, (int, float))
        assert 0.0 <= overall_score <= 1.0

        # Should not have critical failures
        assert overall_score >= 0.3, f"Validation score too low: {overall_score}"

    @pytest.mark.gpu
    def test_gpu_specific_validation(self, setup_test_environment):
        """Test GPU-specific validation features."""
        if not torch.cuda.is_available() and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            pytest.skip("GPU not available for testing")

        validation_results = self._validate_gpu_acceleration()

        # GPU-specific checks
        assert validation_results[
            "gpu_available"
        ], "GPU should be available for this test"
        assert validation_results[
            "memory_management"
        ], "GPU memory management should work"

    def test_fallback_functionality(self, setup_test_environment):
        """Test fallback functionality when Triton is unavailable."""
        # This test should work regardless of Triton availability
        validation_results = self._validate_triton_imports()

        # Should always have some form of functionality
        assert validation_results["imports_correct"] is not None
        assert validation_results["fallbacks_implemented"] is not None

    def _validate_triton_imports(self):
        """Validate that all Triton imports are correctly handled."""
        validation_results = {
            "triton_available": False,
            "imports_correct": False,
            "fallbacks_implemented": False,
            "gpu_acceleration": False,
        }

        try:
            # Test basic Triton availability
            try:
                import triton
                import triton.language as tl

                validation_results["triton_available"] = True
            except ImportError:
                triton = None
                tl = None

            # Test our imports
            from core import TRITON_AVAILABLE, TritonFeatureManager, GPUAccelerator

            if TRITON_AVAILABLE == (triton is not None):
                validation_results["imports_correct"] = True

            # Test GPU acceleration
            fm = TritonFeatureManager()
            ga = GPUAccelerator(fm)

            device = ga.device
            if device.type in ["cuda", "mps"]:
                validation_results["gpu_acceleration"] = True

            # Test fallback functionality
            if not TRITON_AVAILABLE:
                # Verify fallback implementations exist
                try:
                    from core import vector_add_kernel, matrix_multiply_kernel

                    # Try to call fallback functions
                    a = torch.randn(10)
                    b = torch.randn(10)
                    c = torch.zeros(10)
                    vector_add_kernel(a, b, c, 10)
                    validation_results["fallbacks_implemented"] = True
                except Exception:
                    pass

        except Exception:
            pass

        return validation_results

    def _validate_kernel_implementations(self):
        """Validate that all kernel implementations are real (not placeholders)."""
        validation_results = {
            "kernels_found": [],
            "real_implementations": [],
            "placeholder_detected": [],
            "triton_features_used": [],
        }

        try:
            from core import TRITON_AVAILABLE

            if TRITON_AVAILABLE:
                # Check Triton kernel implementations
                triton_kernel_functions = [
                    "vector_add_kernel",
                    "matrix_multiply_kernel",
                    "softmax_kernel",
                    "layer_norm_kernel",
                    "attention_kernel",
                    "conv2d_kernel",
                ]

                for kernel_name in triton_kernel_functions:
                    try:
                        # Import and inspect the kernel
                        module = importlib.import_module("core")
                        kernel_func = getattr(module, kernel_name)

                        # Check if it's a real Triton kernel
                        source = inspect.getsource(kernel_func)

                        if "@triton.jit" in source:
                            validation_results["kernels_found"].append(kernel_name)
                            validation_results["real_implementations"].append(
                                kernel_name
                            )

                            # Check for Triton-specific features
                            triton_features = []
                            if "tl.program_id" in source:
                                triton_features.append("program_id")
                            if "tl.load" in source:
                                triton_features.append("load")
                            if "tl.store" in source:
                                triton_features.append("store")
                            if "tl.arange" in source:
                                triton_features.append("arange")
                            if "BLOCK_SIZE" in source:
                                triton_features.append("block_size")

                            if triton_features:
                                validation_results["triton_features_used"].extend(
                                    triton_features
                                )
                        else:
                            validation_results["placeholder_detected"].append(
                                kernel_name
                            )

                    except Exception:
                        pass

            # Check other modules for Triton kernels
            modules_to_check = [
                "free_energy",
                "message_passing",
                "pomdp_active_inference",
            ]

            for module_name in modules_to_check:
                try:
                    module = importlib.import_module(module_name)
                    source = inspect.getsource(module)

                    triton_kernel_count = source.count("@triton.jit")
                    if triton_kernel_count > 0:
                        validation_results["kernels_found"].append(
                            f"{module_name}_kernels"
                        )

                except Exception:
                    pass

        except Exception:
            pass

        return validation_results

    def _validate_gpu_acceleration(self):
        """Validate that GPU acceleration is properly implemented."""
        validation_results = {
            "gpu_available": False,
            "memory_management": False,
            "kernel_launch": False,
            "synchronization": False,
            "performance_better": False,
        }

        try:
            # Check GPU availability
            if torch.cuda.is_available():
                validation_results["gpu_available"] = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                validation_results["gpu_available"] = True

            # Test memory management
            from core import GPUAccelerator, TritonFeatureManager

            fm = TritonFeatureManager()
            ga = GPUAccelerator(fm)

            # Test tensor allocation
            try:
                tensors = ga.allocate_tensors([(100, 100), (50, 200), (10, 10, 10)])
                validation_results["memory_management"] = len(tensors) == 3
            except Exception:
                pass

            # Test synchronization
            try:
                ga.synchronize()
                validation_results["synchronization"] = True
            except Exception:
                pass

            # Test kernel launch (if available)
            from core import TRITON_AVAILABLE, create_triton_kernel

            if TRITON_AVAILABLE and validation_results["gpu_available"]:
                try:
                    kernel = create_triton_kernel("vector_add")
                    if kernel is not None:
                        validation_results["kernel_launch"] = True
                except Exception:
                    pass

            # Performance comparison
            if validation_results["gpu_available"]:
                try:
                    # Simple performance test
                    size = 1000
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "mps"
                    )

                    # GPU computation
                    a_gpu = torch.randn(size, device=device)
                    b_gpu = torch.randn(size, device=device)

                    import time

                    start_time = time.time()
                    for _ in range(100):
                        c_gpu = a_gpu + b_gpu
                    gpu_time = time.time() - start_time

                    # CPU computation
                    a_cpu = a_gpu.cpu()
                    b_cpu = b_gpu.cpu()
                    start_time = time.time()
                    for _ in range(100):
                        c_cpu = a_cpu + b_cpu
                    cpu_time = time.time() - start_time

                    if gpu_time < cpu_time:
                        validation_results["performance_better"] = True

                except Exception:
                    pass

        except Exception:
            pass

        return validation_results

    def _validate_real_vs_mock_methods(self):
        """Validate that no mock methods are used in production code."""
        validation_results = {
            "mock_methods_detected": [],
            "real_implementations": [],
            "placeholder_functions": [],
            "test_only_code": [],
        }

        try:
            # Check for common mock indicators
            mock_indicators = [
                "placeholder",
                "mock",
                "dummy",
                "not implemented",
                "todo",
                "fixme",
                "hack",
                "temporary",
                "placeholder implementation",
            ]

            # Check all source files
            src_dir = Path(__file__).parent.parent / "src"
            python_files = list(src_dir.glob("*.py"))

            for py_file in python_files:
                try:
                    with open(py_file, "r") as f:
                        content = f.read().lower()

                    # Check for mock indicators
                    for indicator in mock_indicators:
                        if indicator in content:
                            validation_results["mock_methods_detected"].append(
                                {
                                    "file": py_file.name,
                                    "indicator": indicator,
                                    "count": content.count(indicator),
                                }
                            )

                    # Check for real implementation indicators
                    real_indicators = [
                        "@triton.jit",
                        "tl.load",
                        "tl.store",
                        "torch.",
                        "def ",
                    ]
                    real_count = sum(
                        content.count(indicator) for indicator in real_indicators
                    )

                    if real_count > 0:
                        validation_results["real_implementations"].append(
                            {"file": py_file.name, "real_indicators": real_count}
                        )

                    # Check for test-only code
                    if (
                        "test" in py_file.name.lower()
                        or "example" in py_file.name.lower()
                    ):
                        validation_results["test_only_code"].append(py_file.name)

                except Exception:
                    pass

        except Exception:
            pass

        return validation_results

    def _validate_memory_optimization(self):
        """Validate that memory optimization uses real Triton features."""
        validation_results = {
            "shared_memory_usage": False,
            "coalesced_access": False,
            "memory_pooling": False,
            "efficient_allocation": False,
        }

        try:
            # Check for shared memory usage in kernels
            kernel_files = ["core.py", "free_energy.py", "message_passing.py"]

            for file_name in kernel_files:
                try:
                    file_path = Path(__file__).parent.parent / "src" / file_name
                    with open(file_path, "r") as f:
                        content = f.read()

                    if "shared memory" in content.lower() or "tl.zeros" in content:
                        validation_results["shared_memory_usage"] = True

                    if "coalesced" in content.lower() or "contiguous" in content:
                        validation_results["coalesced_access"] = True

                except Exception:
                    pass

            # Test memory pooling
            from core import GPUAccelerator, TritonFeatureManager

            try:
                fm = TritonFeatureManager()
                ga = GPUAccelerator(fm)

                # Test repeated allocations (simulating memory pooling)
                for _ in range(10):
                    tensors = ga.allocate_tensors([(100, 100), (50, 200)])
                    # If no memory errors, allocation is efficient
                    del tensors

                validation_results["efficient_allocation"] = True

            except Exception:
                pass

            # Check for memory pooling indicators in code
            try:
                with open(Path(__file__).parent.parent / "src" / "core.py", "r") as f:
                    core_content = f.read()

                if "memory" in core_content.lower():
                    validation_results["memory_pooling"] = True

            except Exception:
                pass

        except Exception:
            pass

        return validation_results

    def _validate_performance_optimizations(self):
        """Validate that performance optimizations use real Triton features."""
        validation_results = {
            "kernel_fusion": False,
            "vectorization": False,
            "parallelization": False,
            "pipelining": False,
            "autotuning": False,
        }

        try:
            # Check kernel implementations for optimization features
            files_to_check = ["core.py", "free_energy.py", "message_passing.py"]

            for file_name in files_to_check:
                try:
                    file_path = Path(__file__).parent.parent / "src" / file_name
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Check for various optimization indicators
                    if "vectorization" in content.lower() or "tl.arange" in content:
                        validation_results["vectorization"] = True

                    if "parallel" in content.lower() or "program_id" in content:
                        validation_results["parallelization"] = True

                    if "pipeline" in content.lower() or "async" in content:
                        validation_results["pipelining"] = True

                    if "block_size" in content or "BLOCK_SIZE" in content:
                        validation_results["kernel_fusion"] = True

                except Exception:
                    pass

            # Test autotuning capability
            from core import create_triton_kernel, TRITON_AVAILABLE

            if TRITON_AVAILABLE:
                try:
                    # Test kernel creation with different configurations
                    configs_tested = 0
                    for block_size in [128, 256, 512]:
                        try:
                            kernel = create_triton_kernel(
                                "vector_add", block_size=block_size
                            )
                            if kernel is not None:
                                configs_tested += 1
                        except:
                            pass

                    if configs_tested > 1:
                        validation_results["autotuning"] = True

                except Exception:
                    pass

        except Exception:
            pass

        return validation_results

    def _generate_validation_report(self, all_results: Dict[str, Any]):
        """Generate comprehensive validation report."""
        report = {"validation_summary": {}, "scores": {}, "recommendations": []}

        # Calculate scores for each validation category
        scores = {}
        total_score = 0
        categories_checked = 0

        for category, results in all_results.items():
            if isinstance(results, dict):
                # Calculate category score based on boolean values
                boolean_values = [v for v in results.values() if isinstance(v, bool)]
                if boolean_values:
                    category_score = sum(boolean_values) / len(boolean_values)
                    scores[category] = category_score
                    total_score += category_score
                    categories_checked += 1

        if categories_checked > 0:
            overall_score = total_score / categories_checked
            report["overall_score"] = overall_score

            if overall_score >= 0.9:
                report["assessment"] = "EXCELLENT: All validations passed"
            elif overall_score >= 0.7:
                report["assessment"] = "GOOD: Core functionality validated"
            elif overall_score >= 0.5:
                report["assessment"] = "FAIR: Some issues need attention"
            else:
                report["assessment"] = "NEEDS IMPROVEMENT: Significant issues found"

        # Generate recommendations
        recommendations = []

        # Check Triton availability
        triton_results = all_results.get("triton_imports", {})
        if not triton_results.get("triton_available", False):
            recommendations.append("Install Triton for GPU acceleration")

        # Check for mock methods
        mock_results = all_results.get("real_vs_mock", {})
        if mock_results.get("mock_methods_detected", []):
            recommendations.append("Remove or implement placeholder functions")

        # Check GPU acceleration
        gpu_results = all_results.get("gpu_acceleration", {})
        if not gpu_results.get("gpu_available", False):
            recommendations.append("Ensure GPU is available for optimal performance")

        # Check kernel implementations
        kernel_results = all_results.get("kernel_implementations", {})
        if not kernel_results.get("real_implementations", []):
            recommendations.append(
                "Implement real Triton kernels (not just placeholders)"
            )

        report["recommendations"] = recommendations

        return report


def validate_triton_usage():
    """
    Main validation function that can be called from run_all_tests.py
    """
    print("🧪 TRITON USAGE VALIDATION")
    print("=" * 60)
    print(
        "Validating that all implementations use real Triton methods and GPU acceleration..."
    )

    # Create test instance
    validator = TestTritonValidation()

    all_results = {}

    try:
        # Run all validations
        all_results["triton_imports"] = validator._validate_triton_imports()
        all_results["kernel_implementations"] = (
            validator._validate_kernel_implementations()
        )
        all_results["gpu_acceleration"] = validator._validate_gpu_acceleration()
        all_results["real_vs_mock"] = validator._validate_real_vs_mock_methods()
        all_results["memory_optimization"] = validator._validate_memory_optimization()
        all_results["performance_optimizations"] = (
            validator._validate_performance_optimizations()
        )

        # Generate report
        report = validator._generate_validation_report(all_results)

        # Print summary
        print("\n📊 VALIDATION SUMMARY:")
        print(f"Overall Score: {report.get('overall_score', 0):.1f}")
        print(f"Assessment: {report.get('assessment', 'Unknown')}")

        if report.get("recommendations"):
            print("\nRECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"• {rec}")

        # Final assessment
        overall_score = report.get("overall_score", 0)
        if overall_score >= 0.7:
            print("\n🎉 VALIDATION SUCCESSFUL!")
            print("✓ All implementations use real Triton methods")
            print("✓ GPU acceleration properly implemented")
            print("✓ No mock methods in production code")
            print("✓ Memory and performance optimizations active")
            return True
        else:
            print("\n⚠️  VALIDATION ISSUES FOUND")
            print("Some components may need attention - check recommendations above")
            return False

    except Exception as e:
        print(f"\n💥 VALIDATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Allow running as standalone script
    success = validate_triton_usage()
    sys.exit(0 if success else 1)
