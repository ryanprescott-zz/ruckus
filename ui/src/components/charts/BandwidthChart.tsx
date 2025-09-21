import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import type { ChartDataPoint } from '../../utils/chartDataProcessor';

interface BandwidthChartProps {
  data: ChartDataPoint[];
  title?: string;
  chartType?: 'line' | 'bar';
}

const BandwidthChart: React.FC<BandwidthChartProps> = ({
  data,
  title = 'Memory Bandwidth Performance',
  chartType = 'bar'
}) => {
  // Group data by operation type (copy, read, write)
  const groupDataByOperation = () => {
    const grouped: { [key: string]: any } = {};

    data.forEach(point => {
      const size = point.size || point.name;
      const operation = point.operation || 'bandwidth';

      if (!grouped[size]) {
        grouped[size] = { name: size };
      }

      grouped[size][operation] = point.value;
    });

    return Object.values(grouped);
  };

  // Get unique operations for legend
  const getOperations = () => {
    const operations = new Set(data.map(point => point.operation || 'bandwidth'));
    return Array.from(operations);
  };

  const chartData = groupDataByOperation();
  const operations = getOperations();

  // Color palette for different operations
  const colors = {
    copy: '#8884d8',
    read: '#82ca9d',
    write: '#ffc658',
    bandwidth: '#8884d8'
  };

  const formatTooltip = (value: any, name: string) => [
    `${Number(value).toFixed(2)} GB/s`,
    name.charAt(0).toUpperCase() + name.slice(1)
  ];

  const formatYAxis = (value: number) => `${value.toFixed(1)} GB/s`;

  if (chartData.length === 0) {
    return (
      <div className="chart-container">
        <h3>{title}</h3>
        <div className="no-data">
          <p>No bandwidth data available</p>
        </div>
      </div>
    );
  }

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
          {operations.map(operation => (
            <Line
              key={operation}
              type="monotone"
              dataKey={operation}
              stroke={colors[operation as keyof typeof colors] || '#8884d8'}
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
        {operations.map(operation => (
          <Bar
            key={operation}
            dataKey={operation}
            fill={colors[operation as keyof typeof colors] || '#8884d8'}
            radius={[2, 2, 0, 0]}
          />
        ))}
      </BarChart>
    );
  };

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
          Memory bandwidth across different tensor sizes. Higher values indicate better performance.
        </p>
        <div className="chart-stats">
          <span className="stat-item">
            Sizes tested: {chartData.length}
          </span>
          <span className="stat-item">
            Operations: {operations.join(', ')}
          </span>
        </div>
      </div>
    </div>
  );
};

export default BandwidthChart;