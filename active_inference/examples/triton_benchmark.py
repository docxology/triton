#!/usr/bin/env python3
"""
Triton vs PyTorch Performance Benchmark for Apple Silicon

This script benchmarks basic Triton operations against PyTorch equivalents
on Apple Silicon with MPS acceleration. It provides detailed performance
analysis and recommendations for when to use Triton vs PyTorch.

Usage:
    python triton_benchmark.py [--sizes 1000 10000 100000] [--iterations 100]
"""

import sys
import torch
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the active_inference package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "triton_benchmark"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Import from the active_inference package using standardized imports
import active_inference as ai


class TritonBenchmark:
    """Benchmark suite for Triton vs PyTorch operations on Apple Silicon."""

    def __init__(self, device: torch.device, iterations: int = 100):
        self.device = device
        self.iterations = iterations
        self.results = {}

    def benchmark_operation(self, name: str, pytorch_fn, triton_fn, *args) -> Dict[str, Any]:
        """Benchmark a single operation comparing PyTorch vs Triton."""
        print(f"\n⚡ Benchmarking {name}...")

        # Warm up
        for _ in range(10):
            _ = pytorch_fn(*args)
            _ = triton_fn(*args)

        # Benchmark PyTorch
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start_time = time.time()
        pytorch_results = []
        for _ in range(self.iterations):
            result = pytorch_fn(*args)
            pytorch_results.append(result)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        pytorch_time = (time.time() - start_time) * 1000 / self.iterations

        # Benchmark Triton (using ai prefix)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start_time = time.time()
        triton_results = []
        for _ in range(self.iterations):
            result = triton_fn(*args)
            triton_results.append(result)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        triton_time = (time.time() - start_time) * 1000 / self.iterations

        # Calculate speedup
        speedup = pytorch_time / triton_time if triton_time > 0 else 1.0

        # Verify correctness (compare first results)
        if len(pytorch_results) > 0 and len(triton_results) > 0:
            max_diff = torch.max(torch.abs(pytorch_results[0] - triton_results[0])).item()
            correctness = max_diff < 1e-5
        else:
            correctness = False

        result = {
            'operation': name,
            'pytorch_time_ms': pytorch_time,
            'triton_time_ms': triton_time,
            'speedup': speedup,
            'correctness': correctness,
            'iterations': self.iterations,
            'device': str(self.device)
        }

        print(".2f")
        print(".2f")
        print(".2f")
        print(f"Correctness: {'✅ PASS' if correctness else '❌ FAIL'}")

        return result

    def benchmark_vector_operations(self, sizes: List[int]):
        """Benchmark vector operations at different sizes."""
        print("\n🧮 VECTOR OPERATIONS BENCHMARK")
        print("=" * 60)

        results = []

        for size in sizes:
            print(f"\n📏 Testing with size {size:,}")

            # Create test data
            a = torch.randn(size, device=self.device, dtype=torch.float32)
            b = torch.randn(size, device=self.device, dtype=torch.float32)

            # Benchmark vector addition
            result_add = self.benchmark_operation(
                f"Vector Addition (size={size:,})",
                lambda x, y: x + y,
                triton_add_vectors,
                a, b
            )
            results.append(result_add)

            # Benchmark vector multiplication
            result_mul = self.benchmark_operation(
                f"Vector Multiplication (size={size:,})",
                lambda x, y: x * y,
                triton_multiply_vectors,
                a, b
            )
            results.append(result_mul)

            # Benchmark vector sum
            result_sum = self.benchmark_operation(
                f"Vector Sum (size={size:,})",
                lambda x: torch.sum(x),
                triton_vector_sum,
                a
            )
            results.append(result_sum)

        return results

    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            'timestamp': time.time(),
            'device': str(self.device),
            'iterations': self.iterations,
            'results': results,
            'summary': {}
        }

        # Calculate summary statistics
        triton_faster = sum(1 for r in results if r['speedup'] > 1.0)
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        all_correct = all(r['correctness'] for r in results)

        report['summary'] = {
            'total_operations': len(results),
            'triton_faster_count': triton_faster,
            'triton_faster_percentage': triton_faster / len(results) * 100,
            'average_speedup': avg_speedup,
            'all_correct': all_correct,
            'recommendations': self._generate_recommendations(results)
        }

        return report

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []

        # Analyze speedup patterns
        speedups = [r['speedup'] for r in results]
        avg_speedup = sum(speedups) / len(speedups)

        if avg_speedup > 1.2:
            recommendations.append("🎯 Triton provides significant performance benefits for basic operations")
        elif avg_speedup > 0.8:
            recommendations.append("⚖️  Triton and PyTorch have similar performance for basic operations")
        else:
            recommendations.append("🐍 PyTorch may be faster for basic operations on this hardware")

        # Check correctness
        all_correct = all(r['correctness'] for r in results)
        if not all_correct:
            recommendations.append("⚠️  Some operations failed correctness checks - investigate kernel implementations")

        # Device-specific recommendations
        if self.device.type == "mps":
            recommendations.append("🍎 For Apple Silicon, Triton provides reliable basic operations with MPS acceleration")
        elif self.device.type == "cuda":
            recommendations.append("🎮 For CUDA devices, Triton typically provides better performance for complex operations")

        return recommendations

    def save_report(self, report: Dict[str, Any], filename: str = "triton_benchmark_report.json"):
        """Save benchmark report to file."""
        filepath = OUTPUTS_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Report saved to: {filepath}")

    def print_summary(self, report: Dict[str, Any]):
        """Print benchmark summary."""
        summary = report['summary']

        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Device: {report['device']}")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Triton Faster: {summary['triton_faster_count']} ({summary['triton_faster_percentage']:.1f}%)")
        print(".2f")
        print(f"Correctness: {'✅ All PASS' if summary['all_correct'] else '❌ Some FAIL'}")

        print("\n💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")


def main():
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Triton vs PyTorch Benchmark for Apple Silicon")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000],
                       help="Vector sizes to test")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations per benchmark")
    parser.add_argument("--output", type=str, default="triton_benchmark_report.json",
                       help="Output filename for the report")

    args = parser.parse_args()

    print("🚀 TRITON vs PYTORCH BENCHMARK SUITE")
    print("=" * 60)

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    print(f"Test sizes: {args.sizes}")
    print(f"Iterations per test: {args.iterations}")

    try:
        # Run benchmarks
        benchmark = TritonBenchmark(device, args.iterations)
        results = benchmark.benchmark_vector_operations(args.sizes)

        # Generate and save report
        report = benchmark.generate_report(results)
        benchmark.save_report(report, args.output)
        benchmark.print_summary(report)

        print("\n🎉 Benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
