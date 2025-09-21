import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadialBarChart,
  RadialBar
} from 'recharts';
import type { ChartDataPoint } from '../../utils/chartDataProcessor';

interface LLMPerformanceChartProps {
  data: ChartDataPoint[];
  title?: string;
  chartType?: 'bar' | 'pie' | 'radial';
}

const LLMPerformanceChart: React.FC<LLMPerformanceChartProps> = ({
  data,
  title = 'LLM Performance Metrics',
  chartType = 'bar'
}) => {
  // Color palette for different metrics
  const colors = {
    'Inference Time': '#ff7c7c',
    'Throughput': '#82ca9d',
    'Memory Usage': '#8884d8',
    'GPU Utilization': '#ffc658'
  };

  const getColor = (name: string) => {
    return colors[name as keyof typeof colors] || '#8dd1e1';
  };

  // Format different metric types
  const formatValue = (value: number, unit: string) => {
    if (unit === 'seconds') {
      return `${value.toFixed(3)}s`;
    } else if (unit === 'tokens/sec') {
      return `${value.toFixed(1)} tok/s`;
    } else if (unit === 'MB') {
      return `${value.toFixed(0)} MB`;
    } else if (unit === '%') {
      return `${value.toFixed(1)}%`;
    }
    return `${value} ${unit}`;
  };

  const formatTooltip = (value: any, name: string, props: any) => {
    const unit = props.payload?.unit || '';
    return [formatValue(Number(value), unit), name];
  };

  if (data.length === 0) {
    return (
      <div className="chart-container">
        <h3>{title}</h3>
        <div className="no-data">
          <p>No LLM performance data available</p>
        </div>
      </div>
    );
  }

  const renderBarChart = () => (
    <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis
        dataKey="name"
        angle={-45}
        textAnchor="end"
        height={80}
        interval={0}
      />
      <YAxis tickFormatter={(value) => value.toString()} />
      <Tooltip formatter={formatTooltip} />
      <Legend />
      <Bar
        dataKey="value"
        radius={[4, 4, 0, 0]}
        fill="#8884d8"
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={getColor(entry.name)} />
        ))}
      </Bar>
    </BarChart>
  );

  const renderPieChart = () => (
    <PieChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
      <Pie
        data={data as any[]}
        cx="50%"
        cy="50%"
        outerRadius={120}
        fill="#8884d8"
        dataKey="value"
        label={({ name, value, unit }) => `${name}: ${formatValue(Number(value), String(unit))}`}
        labelLine={false}
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={getColor(entry.name)} />
        ))}
      </Pie>
      <Tooltip formatter={formatTooltip} />
    </PieChart>
  );

  const renderRadialChart = () => {
    // Normalize values for radial display (0-100 scale)
    const normalizedData = data.map(item => {
      let normalizedValue = item.value;

      // Normalize based on metric type
      if (item.unit === '%') {
        normalizedValue = item.value; // Already 0-100
      } else if (item.unit === 'seconds') {
        // Invert inference time (lower is better), scale to 0-100
        normalizedValue = Math.max(0, 100 - (item.value * 10));
      } else if (item.unit === 'tokens/sec') {
        // Scale throughput (higher is better)
        normalizedValue = Math.min(100, item.value / 2);
      } else if (item.unit === 'MB') {
        // Memory usage - lower is better for efficiency
        normalizedValue = Math.max(0, 100 - (item.value / 100));
      }

      return {
        ...item,
        normalizedValue: Math.max(0, Math.min(100, normalizedValue))
      };
    });

    return (
      <RadialBarChart
        data={normalizedData}
        startAngle={90}
        endAngle={-270}
        innerRadius="20%"
        outerRadius="80%"
        margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
      >
        <RadialBar
          dataKey="normalizedValue"
          cornerRadius={4}
          fill="#8884d8"
        >
          {normalizedData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={getColor(entry.name)} />
          ))}
        </RadialBar>
        <Legend
          iconSize={18}
          layout="vertical"
          verticalAlign="middle"
          align="right"
          formatter={(value) => {
            const originalData = data.find(d => d.name === value);
            return originalData
              ? `${value}: ${formatValue(originalData.value, originalData.unit)}`
              : value;
          }}
        />
        <Tooltip formatter={formatTooltip} />
      </RadialBarChart>
    );
  };

  // Calculate performance insights
  const calculateInsights = () => {
    const insights: string[] = [];

    const throughputMetric = data.find(d => d.unit === 'tokens/sec');
    const memoryMetric = data.find(d => d.unit === 'MB');
    const utilizationMetric = data.find(d => d.unit === '%');
    const latencyMetric = data.find(d => d.unit === 'seconds');

    if (throughputMetric && throughputMetric.value > 100) {
      insights.push('🚀 High throughput performance (>100 tokens/sec)');
    }

    if (memoryMetric && memoryMetric.value < 4000) {
      insights.push('💾 Efficient memory usage (<4GB)');
    }

    if (utilizationMetric && utilizationMetric.value > 80) {
      insights.push('⚡ High GPU utilization (>80%)');
    }

    if (latencyMetric && latencyMetric.value < 0.1) {
      insights.push('⏱️ Low latency inference (<100ms)');
    }

    return insights;
  };

  const insights = calculateInsights();

  return (
    <div className="chart-container">
      <h3>{title}</h3>
      <div className="chart-wrapper" style={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          {chartType === 'pie' ? renderPieChart() :
           chartType === 'radial' ? renderRadialChart() :
           renderBarChart()}
        </ResponsiveContainer>
      </div>

      <div className="chart-info">
        <p className="chart-description">
          Key performance metrics for LLM inference. Higher throughput and GPU utilization
          with lower latency and memory usage indicate better performance.
        </p>

        {insights.length > 0 && (
          <div className="performance-insights">
            <h4>Performance Insights:</h4>
            <ul>
              {insights.map((insight, index) => (
                <li key={index}>{insight}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="chart-stats">
          <span className="stat-item">
            Metrics: {data.length}
          </span>
          {data.find(d => d.unit === 'tokens/sec') && (
            <span className="stat-item">
              Throughput: {formatValue(
                data.find(d => d.unit === 'tokens/sec')!.value,
                'tokens/sec'
              )}
            </span>
          )}
          {data.find(d => d.unit === 'seconds') && (
            <span className="stat-item">
              Latency: {formatValue(
                data.find(d => d.unit === 'seconds')!.value,
                'seconds'
              )}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default LLMPerformanceChart;