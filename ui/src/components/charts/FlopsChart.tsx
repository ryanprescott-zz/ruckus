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
  LineChart,
  Line
} from 'recharts';
import type { ChartDataPoint } from '../../utils/chartDataProcessor';

interface FlopsChartProps {
  data: ChartDataPoint[];
  title?: string;
  chartType?: 'bar' | 'line';
}

const FlopsChart: React.FC<FlopsChartProps> = ({
  data,
  title = 'Compute Performance (FLOPS)',
  chartType = 'bar'
}) => {
  // Group data by tensor size and precision
  const groupDataBySize = () => {
    const grouped: { [key: string]: any } = {};

    data.forEach(point => {
      const size = point.size || point.name;
      const precision = point.precision || 'Unknown';

      if (!grouped[size]) {
        grouped[size] = { name: size };
      }

      grouped[size][precision] = point.value;
    });

    return Object.values(grouped);
  };

  // Get unique precisions for legend
  const getPrecisions = () => {
    const precisions = new Set(data.map(point => point.precision || 'Unknown'));
    return Array.from(precisions);
  };

  const chartData = groupDataBySize();
  const precisions = getPrecisions();

  // Color palette for different precisions
  const colors = {
    FP32: '#8884d8',
    FP16: '#82ca9d',
    BF16: '#ffc658',
    INT8: '#ff7c7c',
    Unknown: '#8dd1e1'
  };

  const formatTooltip = (value: any, name: string) => [
    `${Number(value).toFixed(2)} TFLOPS`,
    name
  ];

  const formatYAxis = (value: number) => `${value.toFixed(1)} T`;

  if (chartData.length === 0) {
    return (
      <div className="chart-container">
        <h3>{title}</h3>
        <div className="no-data">
          <p>No FLOPS data available</p>
        </div>
      </div>
    );
  }

  // Calculate speedup ratios relative to FP32
  // Calculate speedup ratios relative to FP32
  // const calculateSpeedups = () => {
  //   return chartData.map(item => {
  //     const fp32Value = item.FP32;
  //     const speedupItem = { ...item };

  //     if (fp32Value && fp32Value > 0) {
  //       precisions.forEach(precision => {
  //         if (precision !== 'FP32' && item[precision]) {
  //           speedupItem[`${precision}_speedup`] = item[precision] / fp32Value;
  //         }
  //       });
  //     }

  //     return speedupItem;
  //   });
  // };

  const renderChart = () => {
    if (chartType === 'line') {
      return (
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="name"
            angle={-45}
            textAnchor="end"
            height={80}
            interval={0}
          />
          <YAxis tickFormatter={formatYAxis} />
          <Tooltip formatter={formatTooltip} />
          <Legend />
          {precisions.map(precision => (
            <Line
              key={precision}
              type="monotone"
              dataKey={precision}
              stroke={colors[precision as keyof typeof colors] || '#8884d8'}
              strokeWidth={2}
              dot={{ r: 4 }}
              connectNulls={false}
            />
          ))}
        </LineChart>
      );
    }

    return (
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          angle={-45}
          textAnchor="end"
          height={80}
          interval={0}
        />
        <YAxis tickFormatter={formatYAxis} />
        <Tooltip formatter={formatTooltip} />
        <Legend />
        {precisions.map(precision => (
          <Bar
            key={precision}
            dataKey={precision}
            fill={colors[precision as keyof typeof colors] || '#8884d8'}
            radius={[2, 2, 0, 0]}
          />
        ))}
      </BarChart>
    );
  };

  // Calculate some summary statistics
  const calculateStats = () => {
    const allValues = data.map(d => d.value);
    const maxFLOPS = Math.max(...allValues);
    const avgFLOPS = allValues.reduce((sum, val) => sum + val, 0) / allValues.length;

    return { maxFLOPS, avgFLOPS };
  };

  const stats = calculateStats();

  return (
    <div className="chart-container">
      <h3>{title}</h3>
      <div className="chart-wrapper" style={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          {renderChart()}
        </ResponsiveContainer>
      </div>
      <div className="chart-info">
        <p className="chart-description">
          Computational performance across different precisions and tensor sizes.
          Higher TFLOPS values indicate better performance.
        </p>
        <div className="chart-stats">
          <span className="stat-item">
            Peak Performance: {stats.maxFLOPS.toFixed(2)} TFLOPS
          </span>
          <span className="stat-item">
            Average Performance: {stats.avgFLOPS.toFixed(2)} TFLOPS
          </span>
          <span className="stat-item">
            Precisions: {precisions.join(', ')}
          </span>
        </div>
      </div>
    </div>
  );
};

export default FlopsChart;