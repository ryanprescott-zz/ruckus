import React, { useState, useMemo, useRef } from 'react';
import type { ExperimentResult } from '../types/api';
import { ChartDataProcessor, type ProcessedChartData } from '../utils/chartDataProcessor';
import { ExportService } from '../services/exportService';
import BandwidthChart from './charts/BandwidthChart';
import FlopsChart from './charts/FlopsChart';
import PrecisionChart from './charts/PrecisionChart';
import LLMPerformanceChart from './charts/LLMPerformanceChart';
import './VisualizationTab.css';
import './charts/charts.css';

interface VisualizationTabProps {
  experimentResult: ExperimentResult;
  onClose?: () => void;
}

// Chart type definitions - focused and purpose-built
type ChartType =
  | 'memory-bandwidth' | 'compute-flops' | 'precision-comparison' | 'performance-summary'  // GPU charts
  | 'llm-performance' | 'efficiency-metrics' | 'response-time' | 'resource-usage' | 'llm-summary'  // LLM charts
  | 'compute-precision' | 'compute-performance';  // Compute benchmark charts

const VisualizationTab: React.FC<VisualizationTabProps> = ({ experimentResult, onClose }) => {
  const [selectedChart, setSelectedChart] = useState<ChartType>('memory-bandwidth');
  const [isExporting, setIsExporting] = useState(false);
  const chartRef = useRef<HTMLDivElement>(null);

  // Process experiment result using the chart data processor
  const processedData: ProcessedChartData = useMemo(() => {
    return ChartDataProcessor.processExperimentResult(experimentResult);
  }, [experimentResult]);

  // Get available chart types based on processed data
  const availableCharts: ChartType[] = useMemo(() => {
    const charts: ChartType[] = [];

    console.log('🔍 Available chart data:', Object.keys(processedData.charts));
    console.log('🔍 Processed data type:', processedData.type);

    // GPU/Memory benchmark charts
    if (processedData.charts['memory-bandwidth']) {
      charts.push('memory-bandwidth');
    }
    if (processedData.charts['compute-flops']) {
      charts.push('compute-flops');
    }
    if (processedData.charts['precision-comparison']) {
      charts.push('precision-comparison');
    }
    if (processedData.charts['performance-summary']) {
      charts.push('performance-summary');
    }

    // LLM charts
    if (processedData.charts['llm-performance']) {
      charts.push('llm-performance');
    }
    if (processedData.charts['efficiency-metrics']) {
      charts.push('efficiency-metrics');
    }
    if (processedData.charts['response-time']) {
      charts.push('response-time');
    }
    if (processedData.charts['resource-usage']) {
      charts.push('resource-usage');
    }
    if (processedData.charts['llm-summary']) {
      charts.push('llm-summary');
    }

    // Compute benchmark charts
    if (processedData.charts['compute-precision']) {
      charts.push('compute-precision');
    }
    if (processedData.charts['compute-performance']) {
      charts.push('compute-performance');
    }

    console.log('🔍 Available charts:', charts);
    return charts;
  }, [processedData]);

  // Auto-select first available chart
  React.useEffect(() => {
    if (availableCharts.length > 0 && !availableCharts.includes(selectedChart)) {
      setSelectedChart(availableCharts[0]);
    }
  }, [availableCharts, selectedChart]);

  // Handle export functionality
  const handleExport = async (format: 'png' | 'svg' | 'csv' | 'pdf') => {
    setIsExporting(true);
    try {
      const selectedData = processedData.charts[selectedChart];
      const filename = ExportService.generateFilename(
        `${selectedChart}-chart`,
        experimentResult.experiment_id,
        format
      );

      switch (format) {
        case 'png':
          if (chartRef.current) {
            await ExportService.exportChartAsPNG(chartRef.current, { filename });
          }
          break;

        case 'csv':
          if (selectedData) {
            ExportService.exportChartDataAsCSV(selectedData, { filename });
          }
          break;

        case 'pdf':
          if (chartRef.current) {
            const chartElements: { [key: string]: HTMLElement } = {};
            chartElements[selectedChart] = chartRef.current;
            await ExportService.generatePDFReport(
              experimentResult,
              processedData,
              chartElements,
              { filename }
            );
          }
          break;

        case 'svg':
          // SVG export would need additional implementation
          console.warn('SVG export not yet implemented');
          break;

        default:
          throw new Error(`Unsupported export format: ${format}`);
      }

      console.log(`Successfully exported as ${format}`);
    } catch (error) {
      console.error('Export failed:', error);
      alert(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsExporting(false);
    }
  };

  const renderChartTypeSelector = () => (
    <div className="chart-type-selector">
      <h3>Available Charts</h3>
      <div className="chart-buttons">
        {availableCharts.map(chartType => (
          <button
            key={chartType}
            className={`chart-type-btn ${selectedChart === chartType ? 'active' : ''}`}
            onClick={() => setSelectedChart(chartType)}
          >
            {getChartTypeLabel(chartType)}
          </button>
        ))}
      </div>
    </div>
  );

  const getChartTypeLabel = (type: ChartType): string => {
    switch (type) {
      // GPU benchmark charts - focused and readable
      case 'memory-bandwidth': return '💾 Memory Speed';
      case 'compute-flops': return '⚡ Compute Power';
      case 'precision-comparison': return '🎯 Precision Performance';
      case 'performance-summary': return '📊 Performance Overview';

      // LLM generation charts - focused on user questions
      case 'llm-performance': return '🚀 Generation Speed';
      case 'efficiency-metrics': return '📈 Resource Efficiency';
      case 'response-time': return '⏱️ Response Time';
      case 'resource-usage': return '💾 Memory Usage';
      case 'llm-summary': return '📋 Performance Summary';

      // Compute benchmark charts
      case 'compute-precision': return '🎯 Precision Performance';
      case 'compute-performance': return '⚡ Peak Performance';

      default: return String(type).replace(/-/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase());
    }
  };

  const renderChart = () => {
    const selectedData = processedData.charts[selectedChart];

    if (!selectedData || selectedData.length === 0) {
      return (
        <div className="no-data">
          <p>No data available for {getChartTypeLabel(selectedChart)}</p>
        </div>
      );
    }

    // Render appropriate chart based on type - using focused, readable components
    switch (selectedChart) {
      // GPU Memory Performance
      case 'memory-bandwidth':
        return (
          <BandwidthChart
            data={selectedData}
            title="Memory Bandwidth Performance"
            chartType="bar"
          />
        );

      // GPU Compute Performance
      case 'compute-flops':
        return (
          <FlopsChart
            data={selectedData}
            title="Compute Performance"
            chartType="bar"
          />
        );

      // GPU Precision Comparison
      case 'precision-comparison':
        return (
          <PrecisionChart
            data={selectedData}
            title="Precision Performance Comparison"
            chartType="bar"
            showSpeedup={true}
          />
        );

      // GPU Performance Overview
      case 'performance-summary':
        return (
          <FlopsChart
            data={selectedData}
            title="Performance Summary"
            chartType="bar"
          />
        );

      // LLM Generation Speed
      case 'llm-performance':
        return (
          <LLMPerformanceChart
            data={selectedData}
            title="Token Generation Performance"
            chartType="bar"
          />
        );

      // LLM Resource Efficiency
      case 'efficiency-metrics':
        return (
          <LLMPerformanceChart
            data={selectedData}
            title="Resource Efficiency Metrics"
            chartType="radial"
          />
        );

      // LLM Response Time
      case 'response-time':
        return (
          <LLMPerformanceChart
            data={selectedData}
            title="Response Time Analysis"
            chartType="bar"
          />
        );

      // LLM Memory Usage
      case 'resource-usage':
        return (
          <LLMPerformanceChart
            data={selectedData}
            title="Memory Resource Usage"
            chartType="bar"
          />
        );

      // LLM Performance Summary
      case 'llm-summary':
        return (
          <LLMPerformanceChart
            data={selectedData}
            title="Performance Summary"
            chartType="radial"
          />
        );

      // Compute Benchmark Precision Comparison
      case 'compute-precision':
        return (
          <PrecisionChart
            data={selectedData}
            title="Compute Precision Performance"
            chartType="bar"
            showSpeedup={false}
          />
        );

      // Compute Benchmark Peak Performance
      case 'compute-performance':
        return (
          <FlopsChart
            data={selectedData}
            title="Peak Compute Performance"
            chartType="bar"
          />
        );

      default:
        return (
          <div className="no-data">
            <p>Chart type not implemented: {selectedChart}</p>
          </div>
        );
    }
  };

  const renderExportPanel = () => (
    <div className="export-panel">
      <h4>Export Options</h4>
      <div className="export-buttons">
        <button
          onClick={() => handleExport('png')}
          disabled={isExporting}
          className="export-btn"
        >
          📥 PNG
        </button>
        <button
          onClick={() => handleExport('svg')}
          disabled={isExporting}
          className="export-btn"
        >
          📄 SVG
        </button>
        <button
          onClick={() => handleExport('csv')}
          disabled={isExporting}
          className="export-btn"
        >
          📊 CSV
        </button>
        <button
          onClick={() => handleExport('pdf')}
          disabled={isExporting}
          className="export-btn"
        >
          📑 PDF
        </button>
      </div>
      {isExporting && <div className="export-status">Exporting...</div>}
    </div>
  );

  return (
    <div className="visualization-tab">
      <div className="visualization-header">
        <div className="header-info">
          <h2>Benchmark Visualization</h2>
          <div className="experiment-info">
            <span className="experiment-id">Experiment: {experimentResult.experiment_id}</span>
            <span className="job-id">Job: {experimentResult.job_id}</span>
            <span className="benchmark-type">Type: {processedData.type}</span>
            {processedData.metadata.device && (
              <span className="device-info">Device: {processedData.metadata.device}</span>
            )}
          </div>
        </div>
        {onClose && (
          <button onClick={onClose} className="close-btn" title="Close visualization">
            ✕
          </button>
        )}
      </div>

      <div className="visualization-content">
        <div className="sidebar">
          {renderChartTypeSelector()}
          {renderExportPanel()}
        </div>

        <div className="chart-area" ref={chartRef}>
          {renderChart()}
        </div>
      </div>
    </div>
  );
};

export default VisualizationTab;