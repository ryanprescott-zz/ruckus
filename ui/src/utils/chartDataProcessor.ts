import type { ExperimentResult } from '../types/api';

// Chart data types
export interface ChartDataPoint {
  name: string;
  value: number;
  unit: string;
  category: string;
  size?: string;
  precision?: string;
  operation?: string;
}

export interface ProcessedChartData {
  type: BenchmarkType;
  charts: {
    [chartType: string]: ChartDataPoint[];
  };
  metadata: {
    agentId: string;
    experimentId: string;
    jobId: string;
    timestamp: string;
    device?: string;
  };
}

export type BenchmarkType =
  | 'gpu-benchmark'
  | 'llm-generation'
  | 'memory-benchmark'
  | 'compute-benchmark'
  | 'unknown';

export class ChartDataProcessor {
  /**
   * Detect the type of benchmark from experiment result data
   */
  static detectBenchmarkType(result: ExperimentResult): BenchmarkType {
    const metrics = result.metrics;

    // Chart data processing - now using correct snake_case metric keys

    // Check for LLM generation data (snake_case format)
    if (metrics && (
      'inference_time_seconds' in metrics ||
      'throughput_tokens_per_sec' in metrics ||
      'memory_usage_mb' in metrics ||
      'gpu_utilization_percent' in metrics
    )) {
      return 'llm-generation';
    }

    // Check for GPU/Memory benchmark data - they have the same structure
    if (metrics && (
      'memory_copy_bandwidth_gb_s' in metrics ||
      'memory_read_bandwidth_gb_s' in metrics ||
      'memory_write_bandwidth_gb_s' in metrics ||
      'compute_fp32_tflops' in metrics ||
      'precision_fp32_gops' in metrics ||
      'precision_fp16_gops' in metrics
    )) {
      return 'gpu-benchmark';
    }

    // Check for memory benchmark data
    if (metrics && ('memory_test' in metrics || 'bandwidth_test' in metrics)) {
      return 'memory-benchmark';
    }

    // Check for compute benchmark data (using actual key format)
    if (metrics && (
      'matrix_sizes_tested' in metrics ||
      'precision_types_tested' in metrics ||
      'tensor_ops_included' in metrics ||
      'peak_compute_tflops' in metrics ||
      'PrecisionType.FP32_tflops' in metrics ||
      'PrecisionType.FP16_tflops' in metrics
    )) {
      return 'compute-benchmark';
    }

    return 'unknown';
  }

  /**
   * Process experiment result into chart-ready data
   */
  static processExperimentResult(result: ExperimentResult): ProcessedChartData {
    const benchmarkType = this.detectBenchmarkType(result);
    const charts: { [chartType: string]: ChartDataPoint[] } = {};

    switch (benchmarkType) {
      case 'gpu-benchmark':
        Object.assign(charts, this.processGPUBenchmarkData(result.metrics));
        break;
      case 'llm-generation':
        Object.assign(charts, this.processLLMGenerationData(result.metrics));
        break;
      case 'memory-benchmark':
        Object.assign(charts, this.processMemoryBenchmarkData(result.metrics));
        break;
      case 'compute-benchmark':
        Object.assign(charts, this.processComputeBenchmarkData(result.metrics));
        break;
    }

    return {
      type: benchmarkType,
      charts,
      metadata: {
        agentId: result.agent_id,
        experimentId: result.experiment_id,
        jobId: result.job_id,
        timestamp: result.completed_at || result.started_at,
        device: this.extractDeviceInfo(result.metrics)
      }
    };
  }

  /**
   * Process GPU benchmark data into charts
   */
  private static processGPUBenchmarkData(metrics: any): { [chartType: string]: ChartDataPoint[] } {
    const charts: { [chartType: string]: ChartDataPoint[] } = {};

    // 1. Memory Bandwidth Chart - answers "How fast is my memory?"
    const bandwidthData: ChartDataPoint[] = [];

    if (metrics.memory_copy_bandwidth_gb_s) {
      bandwidthData.push({
        name: 'Memory Copy',
        value: metrics.memory_copy_bandwidth_gb_s,
        unit: 'GB/s',
        category: 'memory-bandwidth',
        operation: 'copy'
      });
    }

    if (metrics.memory_read_bandwidth_gb_s) {
      bandwidthData.push({
        name: 'Memory Read',
        value: metrics.memory_read_bandwidth_gb_s,
        unit: 'GB/s',
        category: 'memory-bandwidth',
        operation: 'read'
      });
    }

    if (metrics.memory_write_bandwidth_gb_s) {
      bandwidthData.push({
        name: 'Memory Write',
        value: metrics.memory_write_bandwidth_gb_s,
        unit: 'GB/s',
        category: 'memory-bandwidth',
        operation: 'write'
      });
    }

    if (bandwidthData.length > 0) {
      charts['memory-bandwidth'] = bandwidthData;
    }

    // 2. Compute Performance Chart - answers "How much compute power do I have?"
    const computeData: ChartDataPoint[] = [];

    if (metrics.compute_fp32_tflops) {
      computeData.push({
        name: 'FP32 Compute',
        value: metrics.compute_fp32_tflops,
        unit: 'TFLOPS',
        category: 'compute-performance',
        precision: 'FP32'
      });
    }

    if (computeData.length > 0) {
      charts['compute-flops'] = computeData;
    }

    // 3. Precision Efficiency Chart - answers "Which precision should I use?"
    // Only show if we have multiple precisions to compare
    const precisionData: ChartDataPoint[] = [];

    if (metrics.precision_fp32_gops && metrics.precision_fp32_gops > 0) {
      precisionData.push({
        name: 'FP32 Precision',
        value: metrics.precision_fp32_gops,
        unit: 'GOPS',
        category: 'precision-throughput',
        precision: 'FP32'
      });
    }

    if (metrics.precision_fp16_gops && metrics.precision_fp16_gops > 0) {
      precisionData.push({
        name: 'FP16 Precision',
        value: metrics.precision_fp16_gops,
        unit: 'GOPS',
        category: 'precision-throughput',
        precision: 'FP16'
      });
    }

    // Only add precision chart if we have data to compare
    if (precisionData.length > 0) {
      charts['precision-comparison'] = precisionData;
    }

    // 4. Performance Summary - answers "What's my overall performance profile?"
    // Create a normalized performance overview for quick insights
    const summaryData: ChartDataPoint[] = [];

    // Add key performance indicators as percentages/scores
    if (metrics.memory_read_bandwidth_gb_s) {
      // Normalize read bandwidth (assuming ~100 GB/s as good performance baseline)
      const score = Math.min(100, (metrics.memory_read_bandwidth_gb_s / 100) * 100);
      summaryData.push({
        name: 'Memory Speed',
        value: score,
        unit: '% efficiency',
        category: 'performance-summary'
      });
    }

    if (metrics.compute_fp32_tflops) {
      // Normalize compute performance (assuming ~50 TFLOPS as good baseline)
      const score = Math.min(100, (metrics.compute_fp32_tflops / 50) * 100);
      summaryData.push({
        name: 'Compute Power',
        value: score,
        unit: '% efficiency',
        category: 'performance-summary'
      });
    }

    if (summaryData.length > 0) {
      charts['performance-summary'] = summaryData;
    }

    return charts;
  }

  /**
   * Process LLM generation data into charts
   */
  private static processLLMGenerationData(metrics: any): { [chartType: string]: ChartDataPoint[] } {
    const charts: { [chartType: string]: ChartDataPoint[] } = {};

    // 1. Performance Chart - answers "How fast can I generate tokens?"
    if (metrics.throughput_tokens_per_sec) {
      charts['llm-performance'] = [{
        name: 'Token Generation Speed',
        value: metrics.throughput_tokens_per_sec,
        unit: 'tokens/sec',
        category: 'llm-throughput'
      }];
    }

    // 2. Efficiency Chart - answers "Am I using resources efficiently?"
    const efficiencyData: ChartDataPoint[] = [];

    if (metrics.gpu_utilization_percent) {
      efficiencyData.push({
        name: 'GPU Utilization',
        value: metrics.gpu_utilization_percent,
        unit: '%',
        category: 'llm-efficiency'
      });
    }

    // Calculate tokens per MB if we have both metrics
    if (metrics.throughput_tokens_per_sec && metrics.memory_usage_mb) {
      const tokensPerMB = metrics.throughput_tokens_per_sec / (metrics.memory_usage_mb / 1000); // tokens per GB
      efficiencyData.push({
        name: 'Memory Efficiency',
        value: tokensPerMB,
        unit: 'tokens/sec per GB',
        category: 'llm-efficiency'
      });
    }

    if (efficiencyData.length > 0) {
      charts['efficiency-metrics'] = efficiencyData;
    }

    // 3. Responsiveness Chart - answers "How quickly does it respond?"
    if (metrics.inference_time_seconds) {
      charts['response-time'] = [{
        name: 'Response Time',
        value: metrics.inference_time_seconds,
        unit: 'seconds',
        category: 'llm-latency'
      }];
    }

    // 4. Resource Usage Chart - answers "What's the resource footprint?"
    if (metrics.memory_usage_mb) {
      charts['resource-usage'] = [{
        name: 'Memory Usage',
        value: metrics.memory_usage_mb / 1024, // Convert to GB for readability
        unit: 'GB',
        category: 'llm-resources'
      }];
    }

    // 5. Performance Summary - overall health check
    const summaryData: ChartDataPoint[] = [];

    // Performance score based on throughput (assuming 100 tokens/sec as good baseline)
    if (metrics.throughput_tokens_per_sec) {
      const perfScore = Math.min(100, (metrics.throughput_tokens_per_sec / 100) * 100);
      summaryData.push({
        name: 'Performance Score',
        value: perfScore,
        unit: '% of target',
        category: 'llm-summary'
      });
    }

    // Efficiency score based on GPU utilization
    if (metrics.gpu_utilization_percent) {
      summaryData.push({
        name: 'Efficiency Score',
        value: metrics.gpu_utilization_percent,
        unit: '% GPU used',
        category: 'llm-summary'
      });
    }

    // Responsiveness score (lower latency = higher score)
    if (metrics.inference_time_seconds) {
      const responseScore = Math.max(0, 100 - (metrics.inference_time_seconds * 100)); // penalty for high latency
      summaryData.push({
        name: 'Responsiveness Score',
        value: Math.max(0, responseScore),
        unit: '% responsiveness',
        category: 'llm-summary'
      });
    }

    if (summaryData.length > 0) {
      charts['llm-summary'] = summaryData;
    }

    return charts;
  }

  /**
   * Process memory benchmark data into charts
   */
  private static processMemoryBenchmarkData(data: any): { [chartType: string]: ChartDataPoint[] } {
    const charts: { [chartType: string]: ChartDataPoint[] } = {};

    // TODO: Implement when memory benchmark structure is defined
    console.log('Memory benchmark data processing not yet implemented', data);

    return charts;
  }

  /**
   * Process compute benchmark data into charts
   */
  private static processComputeBenchmarkData(metrics: any): { [chartType: string]: ChartDataPoint[] } {
    const charts: { [chartType: string]: ChartDataPoint[] } = {};

    // Precision performance comparison (using actual key format)
    const precisionData: ChartDataPoint[] = [];

    if (metrics['PrecisionType.FP32_tflops']) {
      precisionData.push({
        name: 'FP32',
        value: metrics['PrecisionType.FP32_tflops'],
        unit: 'TFLOPS',
        category: 'compute-precision',
        precision: 'FP32'
      });
    }

    if (metrics['PrecisionType.FP16_tflops']) {
      precisionData.push({
        name: 'FP16',
        value: metrics['PrecisionType.FP16_tflops'],
        unit: 'TFLOPS',
        category: 'compute-precision',
        precision: 'FP16'
      });
    }

    if (precisionData.length > 0) {
      charts['compute-precision'] = precisionData;
      console.log('✅ Added compute-precision chart with', precisionData.length, 'data points');
    }

    // Overall compute performance
    const computeData: ChartDataPoint[] = [];

    if (metrics.peak_compute_tflops) {
      computeData.push({
        name: 'Peak Performance',
        value: metrics.peak_compute_tflops,
        unit: 'TFLOPS',
        category: 'compute-performance'
      });
    }

    if (computeData.length > 0) {
      charts['compute-performance'] = computeData;
      console.log('✅ Added compute-performance chart with', computeData.length, 'data points');
    }

    console.log('🔍 Final compute benchmark charts:', Object.keys(charts));
    return charts;
  }

  /**
   * Extract device information from benchmark data
   */
  private static extractDeviceInfo(metrics: any): string | undefined {
    // Look for explicit device info in metrics (snake_case)
    if (metrics?.gpu_device) {
      return metrics.gpu_device;
    }

    // Try to infer from metrics structure
    if (metrics?.memory_copy_bandwidth_gb_s || metrics?.compute_fp32_tflops) {
      return 'GPU'; // GPU benchmark detected
    }

    return undefined;
  }

  /**
   * Normalize chart data for comparison between different results
   */
  static normalizeForComparison(
    results: ProcessedChartData[],
    normalizationMethod: 'baseline' | 'percentage' | 'zscore' = 'baseline'
  ): ProcessedChartData[] {
    // Find common chart types across all results
    const commonChartTypes = this.findCommonChartTypes(results);

    return results.map(result => {
      const normalizedCharts: { [chartType: string]: ChartDataPoint[] } = {};

      commonChartTypes.forEach(chartType => {
        const chartData = result.charts[chartType];
        if (chartData) {
          normalizedCharts[chartType] = this.normalizeChartData(
            chartData,
            results.map(r => r.charts[chartType]).filter(Boolean),
            normalizationMethod
          );
        }
      });

      return {
        ...result,
        charts: normalizedCharts
      };
    });
  }

  /**
   * Find chart types that exist across all results
   */
  private static findCommonChartTypes(results: ProcessedChartData[]): string[] {
    if (results.length === 0) return [];

    const firstResultChartTypes = Object.keys(results[0].charts);

    return firstResultChartTypes.filter(chartType =>
      results.every(result => chartType in result.charts)
    );
  }

  /**
   * Normalize a single chart's data
   */
  private static normalizeChartData(
    data: ChartDataPoint[],
    allData: ChartDataPoint[][],
    method: 'baseline' | 'percentage' | 'zscore'
  ): ChartDataPoint[] {
    switch (method) {
      case 'baseline': {
        // Use the minimum value as baseline (1.0x)
        const allValues = allData.flat().map(d => d.value);
        const baseline = Math.min(...allValues);

        return data.map(point => ({
          ...point,
          value: baseline > 0 ? point.value / baseline : point.value,
          unit: 'x speedup'
        }));
      }

      case 'percentage': {
        // Convert to percentage of maximum value
        const allValues = allData.flat().map(d => d.value);
        const maximum = Math.max(...allValues);

        return data.map(point => ({
          ...point,
          value: maximum > 0 ? (point.value / maximum) * 100 : 0,
          unit: '% of max'
        }));
      }

      case 'zscore': {
        // Z-score normalization
        const allValues = allData.flat().map(d => d.value);
        const mean = allValues.reduce((sum, val) => sum + val, 0) / allValues.length;
        const stdDev = Math.sqrt(
          allValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / allValues.length
        );

        return data.map(point => ({
          ...point,
          value: stdDev > 0 ? (point.value - mean) / stdDev : 0,
          unit: 'σ'
        }));
      }

      default:
        return data;
    }
  }

  /**
   * Generate summary statistics for chart data
   */
  static generateSummaryStats(data: ChartDataPoint[]): {
    mean: number;
    median: number;
    min: number;
    max: number;
    stdDev: number;
  } {
    const values = data.map(d => d.value).sort((a, b) => a - b);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const median = values.length % 2 === 0
      ? (values[values.length / 2 - 1] + values[values.length / 2]) / 2
      : values[Math.floor(values.length / 2)];
    const min = values[0];
    const max = values[values.length - 1];
    const stdDev = Math.sqrt(
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
    );

    return { mean, median, min, max, stdDev };
  }
}