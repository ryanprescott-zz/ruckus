"""GPU benchmarking utilities for memory bandwidth and FLOPS testing."""

import logging
import time
from typing import Dict, List, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)


class GPUBenchmark:
    """Comprehensive GPU benchmarking for memory bandwidth and FLOPS across precisions."""
    
    def __init__(self):
        self.torch = None
        self.device = None
        
    async def initialize(self, device_index: int = 0) -> bool:
        """Initialize the benchmark with the specified GPU device."""
        try:
            import torch
            self.torch = torch
            
            if torch.cuda.is_available():
                self.device = f'cuda:{device_index}'
                torch.cuda.set_device(device_index)
                # Warm up the GPU
                _ = torch.randn(100, 100, device=self.device)
                torch.cuda.synchronize()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
                
            logger.info(f"GPU benchmark initialized on device: {self.device}")
            return True
            
        except ImportError:
            logger.error("PyTorch not available for GPU benchmarking")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize GPU benchmark: {e}")
            return False
    
    async def run_comprehensive_benchmark(self, available_memory_mb: int) -> Dict:
        """Run comprehensive GPU benchmarks adapted to available VRAM."""
        if not self.torch or not self.device:
            raise RuntimeError("Benchmark not initialized. Call initialize() first.")
        
        logger.info(f"Running comprehensive benchmark with {available_memory_mb}MB available memory")
        
        results = {
            "memory_bandwidth": await self.benchmark_memory_bandwidth(available_memory_mb),
            "compute_flops": await self.benchmark_compute_flops(available_memory_mb),
            "precision_performance": await self.benchmark_precision_performance(available_memory_mb),
            "tensor_sizes_tested": self._calculate_tensor_sizes(available_memory_mb),
            "benchmark_device": str(self.device)
        }
        
        return results
    
    def _calculate_tensor_sizes(self, available_memory_mb: int) -> List[Tuple[int, int]]:
        """Calculate tensor sizes to test based on available VRAM."""
        # Use fractions of available memory to create a scaling curve
        memory_bytes = available_memory_mb * 1024 * 1024
        
        # Test sizes from small to large (10MB to 80% of available memory)
        test_sizes = []
        
        # Start with small sizes for consistency testing
        base_sizes = [
            (512, 512),      # ~1MB for FP32
            (1024, 1024),    # ~4MB for FP32  
            (2048, 2048),    # ~16MB for FP32
        ]
        
        # Add adaptive sizes based on available memory
        if available_memory_mb > 1000:  # > 1GB
            base_sizes.extend([
                (4096, 4096),    # ~64MB for FP32
                (8192, 8192),    # ~256MB for FP32
            ])
            
        if available_memory_mb > 4000:  # > 4GB
            base_sizes.extend([
                (16384, 16384),  # ~1GB for FP32
            ])
            
        # Calculate largest size we can safely test (use ~50% of available memory)
        max_elements = (memory_bytes * 0.5) // 4  # 4 bytes per FP32
        max_side = int((max_elements ** 0.5))
        
        # Round down to nearest power of 2 for consistency
        max_side_pow2 = 2 ** int(max_side.bit_length() - 1)
        if max_side_pow2 > 16384:
            base_sizes.append((max_side_pow2, max_side_pow2))
        
        return base_sizes
    
    async def benchmark_memory_bandwidth(self, available_memory_mb: int) -> Dict:
        """Benchmark memory bandwidth with tensor copy operations."""
        logger.info("Starting memory bandwidth benchmark with DIVISION BY ZERO PROTECTION")
        print(f"DEBUG: Memory bandwidth benchmark starting with {available_memory_mb}MB available")
        
        tensor_sizes = self._calculate_tensor_sizes(available_memory_mb)
        bandwidth_results = {}
        
        for size_tuple in tensor_sizes:
            rows, cols = size_tuple
            size_name = f"{rows}x{cols}"
            
            try:
                # Test memory bandwidth with different operations
                results = {}
                
                # Memory allocation and copy bandwidth
                results['copy_bandwidth_gb_s'] = await self._test_memory_copy_bandwidth(rows, cols)
                
                # Memory write bandwidth  
                results['write_bandwidth_gb_s'] = await self._test_memory_write_bandwidth(rows, cols)
                
                # Memory read bandwidth
                results['read_bandwidth_gb_s'] = await self._test_memory_read_bandwidth(rows, cols)
                
                bandwidth_results[size_name] = results
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at size {size_name}, skipping larger sizes")
                    break
                else:
                    logger.error(f"Error testing size {size_name}: {e}")
        
        return bandwidth_results
    
    async def _test_memory_copy_bandwidth(self, rows: int, cols: int) -> float:
        """Test memory copy bandwidth."""
        # Create source tensor on CPU
        cpu_tensor = self.torch.randn(rows, cols, dtype=self.torch.float32)
        tensor_size_bytes = cpu_tensor.numel() * 4  # 4 bytes per float32

        # Warm up
        for _ in range(3):
            gpu_tensor = cpu_tensor.to(self.device)
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        # Adaptive benchmarking - ensure minimum duration
        min_duration = 0.01  # Minimum 10ms duration
        num_iterations = max(10, 1000 // max(1, rows // 1024))  # Adapt iterations to size

        # Run initial benchmark to estimate timing
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            gpu_tensor = cpu_tensor.to(self.device)
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        end_time = time.perf_counter()
        initial_elapsed = end_time - start_time

        # If too fast, increase iterations to reach minimum duration
        if initial_elapsed < min_duration:
            if initial_elapsed > 0:
                scaling_factor = max(2, int(min_duration / initial_elapsed))
            else:
                scaling_factor = 100  # Aggressive scaling if timing is too precise
            num_iterations *= scaling_factor

            # Re-run with adjusted iterations
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(num_iterations):
                gpu_tensor = cpu_tensor.to(self.device)
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:
            elapsed_time = initial_elapsed

        total_bytes = tensor_size_bytes * num_iterations

        # Prevent division by zero - if elapsed time is too small, estimate minimum bandwidth
        if elapsed_time <= 0:
            logger.warning(f"Memory copy elapsed time too small ({elapsed_time}s), using fallback")
            bandwidth_gb_s = 0.1  # Conservative fallback bandwidth in GB/s
        else:
            bandwidth_gb_s = (total_bytes / elapsed_time) / (1024**3)

        return bandwidth_gb_s
    
    async def _test_memory_write_bandwidth(self, rows: int, cols: int) -> float:
        """Test memory write bandwidth using tensor operations."""
        num_iterations = max(10, 1000 // max(1, rows // 1024))

        # Create tensors on device
        tensor_a = self.torch.randn(rows, cols, device=self.device, dtype=self.torch.float32)
        tensor_size_bytes = tensor_a.numel() * 4

        # Warm up
        for _ in range(3):
            tensor_a.fill_(1.0)
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        # Adaptive benchmarking - ensure minimum duration
        min_duration = 0.01  # Minimum 10ms duration

        # Run initial benchmark to estimate timing
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        start_time = time.perf_counter()

        for i in range(num_iterations):
            tensor_a.fill_(float(i))
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        end_time = time.perf_counter()
        initial_elapsed = end_time - start_time

        # If too fast, increase iterations to reach minimum duration
        if initial_elapsed < min_duration:
            if initial_elapsed > 0:
                scaling_factor = max(2, int(min_duration / initial_elapsed))
            else:
                scaling_factor = 100  # Aggressive scaling if timing is too precise
            num_iterations *= scaling_factor

            # Re-run with adjusted iterations
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            start_time = time.perf_counter()

            for i in range(num_iterations):
                tensor_a.fill_(float(i % 100))  # Cycle through values to prevent optimization
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:
            elapsed_time = initial_elapsed

        total_bytes = tensor_size_bytes * num_iterations

        # Prevent division by zero - if elapsed time is too small, estimate minimum bandwidth
        if elapsed_time <= 0:
            logger.warning(f"Memory write elapsed time too small ({elapsed_time}s), using fallback")
            bandwidth_gb_s = 0.1  # Conservative fallback bandwidth in GB/s
        else:
            bandwidth_gb_s = (total_bytes / elapsed_time) / (1024**3)

        return bandwidth_gb_s
    
    async def _test_memory_read_bandwidth(self, rows: int, cols: int) -> float:
        """Test memory read bandwidth using reduction operations."""
        num_iterations = max(10, 500 // max(1, rows // 1024))

        tensor_a = self.torch.randn(rows, cols, device=self.device, dtype=self.torch.float32)
        tensor_size_bytes = tensor_a.numel() * 4

        # Warm up
        for _ in range(3):
            _ = tensor_a.sum()
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        # Adaptive benchmarking - ensure minimum duration
        min_duration = 0.01  # Minimum 10ms duration

        # Run initial benchmark to estimate timing
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            result = tensor_a.sum()
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        end_time = time.perf_counter()
        initial_elapsed = end_time - start_time

        # If too fast, increase iterations to reach minimum duration
        if initial_elapsed < min_duration:
            if initial_elapsed > 0:
                scaling_factor = max(2, int(min_duration / initial_elapsed))
            else:
                scaling_factor = 100  # Aggressive scaling if timing is too precise
            num_iterations *= scaling_factor

            # Re-run with adjusted iterations
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(num_iterations):
                result = tensor_a.sum()
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:
            elapsed_time = initial_elapsed

        total_bytes = tensor_size_bytes * num_iterations

        # Prevent division by zero - if elapsed time is too small, estimate minimum bandwidth
        if elapsed_time <= 0:
            logger.warning(f"Memory read elapsed time too small ({elapsed_time}s), using fallback")
            bandwidth_gb_s = 0.1  # Conservative fallback bandwidth in GB/s
        else:
            bandwidth_gb_s = (total_bytes / elapsed_time) / (1024**3)

        return bandwidth_gb_s
    
    async def benchmark_compute_flops(self, available_memory_mb: int) -> Dict:
        """Benchmark compute FLOPS with matrix multiplication."""
        logger.info("Starting compute FLOPS benchmark")
        
        tensor_sizes = self._calculate_tensor_sizes(available_memory_mb)
        flops_results = {}
        
        for size_tuple in tensor_sizes:
            rows, cols = size_tuple
            size_name = f"{rows}x{cols}"
            
            try:
                # Matrix multiplication FLOPS test
                flops_fp32 = await self._test_matmul_flops(rows, cols, self.torch.float32)
                
                result = {"fp32_tflops": flops_fp32}
                
                # Test other precisions if available
                if self.device.startswith('cuda'):
                    # Test FP16 if supported
                    try:
                        flops_fp16 = await self._test_matmul_flops(rows, cols, self.torch.float16)
                        result["fp16_tflops"] = flops_fp16
                    except:
                        pass
                    
                    # Test BF16 if supported  
                    try:
                        flops_bf16 = await self._test_matmul_flops(rows, cols, self.torch.bfloat16)
                        result["bf16_tflops"] = flops_bf16
                    except:
                        pass
                
                flops_results[size_name] = result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at size {size_name}, skipping larger sizes")
                    break
                else:
                    logger.error(f"Error testing FLOPS for size {size_name}: {e}")
        
        return flops_results
    
    async def _test_matmul_flops(self, size: int, size2: int, dtype) -> float:
        """Test matrix multiplication FLOPS for a given precision."""
        # Create matrices
        A = self.torch.randn(size, size2, device=self.device, dtype=dtype)
        B = self.torch.randn(size2, size, device=self.device, dtype=dtype)

        # Warm up
        for _ in range(3):
            _ = self.torch.mm(A, B)
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        # Adaptive benchmarking - ensure minimum duration
        min_duration = 0.01  # Minimum 10ms duration
        num_iterations = max(5, 100 // max(1, size // 2048))

        # Run initial benchmark to estimate timing
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            C = self.torch.mm(A, B)
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        end_time = time.perf_counter()
        initial_elapsed = end_time - start_time

        # If too fast, increase iterations to reach minimum duration
        if initial_elapsed < min_duration:
            if initial_elapsed > 0:
                scaling_factor = max(2, int(min_duration / initial_elapsed))
            else:
                scaling_factor = 100  # Aggressive scaling if timing is too precise
            num_iterations *= scaling_factor

            # Re-run with adjusted iterations
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(num_iterations):
                C = self.torch.mm(A, B)
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:
            elapsed_time = initial_elapsed

        # Calculate FLOPS
        # Matrix multiplication: A(m,k) * B(k,n) requires 2*m*k*n operations
        flops_per_matmul = 2 * size * size2 * size
        total_flops = flops_per_matmul * num_iterations

        # Prevent division by zero
        if elapsed_time <= 0:
            logger.warning(f"Matrix multiply elapsed time too small ({elapsed_time}s), using fallback")
            tflops = 0.01  # Conservative fallback TFLOPS
        else:
            tflops = (total_flops / elapsed_time) / 1e12

        return tflops
    
    async def benchmark_precision_performance(self, available_memory_mb: int) -> Dict:
        """Benchmark relative performance across different precisions."""
        logger.info("Starting precision performance benchmark")
        
        # Use a moderate size for precision comparison
        test_size = min(2048, int((available_memory_mb * 1024 * 1024 * 0.1) ** 0.5))
        
        precisions_to_test = [
            ("fp32", self.torch.float32),
            ("fp16", self.torch.float16),
        ]
        
        # Add more precisions for CUDA devices
        if self.device.startswith('cuda'):
            precisions_to_test.append(("bf16", self.torch.bfloat16))
        
        results = {}
        
        for precision_name, dtype in precisions_to_test:
            try:
                # Test throughput for this precision
                throughput = await self._test_precision_throughput(test_size, dtype)
                results[precision_name] = {
                    "throughput_gops": throughput,
                    "relative_speedup": None  # Will be calculated after
                }
            except Exception as e:
                logger.warning(f"Failed to test {precision_name}: {e}")
        
        # Calculate relative speedups (compared to FP32)
        if "fp32" in results:
            fp32_throughput = results["fp32"]["throughput_gops"]
            if fp32_throughput > 0:
                for precision_name in results:
                    if precision_name != "fp32":
                        speedup = results[precision_name]["throughput_gops"] / fp32_throughput
                        results[precision_name]["relative_speedup"] = speedup
            else:
                logger.warning(f"FP32 throughput is zero ({fp32_throughput}), cannot calculate relative speedups")
                for precision_name in results:
                    if precision_name != "fp32":
                        results[precision_name]["relative_speedup"] = 1.0  # Default to no speedup
        
        return results
    
    async def _test_precision_throughput(self, size: int, dtype) -> float:
        """Test computational throughput for a specific precision."""
        A = self.torch.randn(size, size, device=self.device, dtype=dtype)
        B = self.torch.randn(size, size, device=self.device, dtype=dtype)

        # Warm up
        for _ in range(3):
            C = A @ B  # Matrix multiplication
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        # Adaptive benchmarking - ensure minimum duration
        min_duration = 0.01  # Minimum 10ms duration
        num_iterations = max(10, 200 // max(1, size // 1024))

        # Run initial benchmark to estimate timing
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            C = A @ B
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()

        end_time = time.perf_counter()
        initial_elapsed = end_time - start_time

        # If too fast, increase iterations to reach minimum duration
        if initial_elapsed < min_duration:
            if initial_elapsed > 0:
                scaling_factor = max(2, int(min_duration / initial_elapsed))
            else:
                scaling_factor = 100  # Aggressive scaling if timing is too precise
            num_iterations *= scaling_factor

            # Re-run with adjusted iterations
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(num_iterations):
                C = A @ B
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:
            elapsed_time = initial_elapsed

        # Calculate throughput in GOPS (Giga Operations Per Second)
        ops_per_matmul = 2 * size * size * size  # Matrix multiply operations
        total_ops = ops_per_matmul * num_iterations

        # Prevent division by zero
        if elapsed_time <= 0:
            logger.warning(f"Precision throughput elapsed time too small ({elapsed_time}s), using fallback")
            gops = 0.01  # Conservative fallback GOPS
        else:
            gops = (total_ops / elapsed_time) / 1e9

        return gops