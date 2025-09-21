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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import type { ChartDataPoint } from '../../utils/chartDataProcessor';

interface PrecisionChartProps {
  data: ChartDataPoint[];
  title?: string;
  chartType?: 'bar' | 'radar';
  showSpeedup?: boolean;
}

const PrecisionChart: React.FC<PrecisionChartProps> = ({
  data,
  title = 'Precision Performance Comparison',
  chartType = 'bar',
  showSpeedup = true
}) => {
  // Separate throughput and speedup data
  const throughputData = data.filter(d => d.category === 'precision-throughput');
  const speedupData = data.filter(d => d.category === 'precision-speedup');

  // Combine throughput and speedup data for display
  const combineData = () => {
    const combined: { [precision: string]: any } = {};

    throughputData.forEach(point => {
      const precision = point.precision || point.name;
      if (!combined[precision]) {
        combined[precision] = { name: precision };
      }
      combined[precision].throughput = point.value;
    });

    speedupData.forEach(point => {
      const precision = point.precision || point.name;
      if (!combined[precision]) {
        combined[precision] = { name: precision };
      }
      combined[precision].speedup = point.value;
    });

    return Object.values(combined);
  };

  const chartData = combineData();

  // Color palette for different precisions
  const colors = {
    FP32: '#8884d8',
    FP16: '#82ca9d',
    BF16: '#ffc658',
    INT8: '#ff7c7c'
  };

  const getColor = (precision: string) => {
    return colors[precision as keyof typeof colors] || '#8dd1e1';
  };

  // const formatTooltipThroughput = (value: any) => [`${Number(value).toFixed(2)} GOPS`, 'Throughput'];
  // const formatTooltipSpeedup = (value: any) => [`${Number(value).toFixed(2)}x`, 'Speedup vs FP32'];

  if (chartData.length === 0) {
    return (
      <div className="chart-container">
        <h3>{title}</h3>
        <div className="no-data">
          <p>No precision comparison data available</p>
        </div>
      </div>
    );
  }

  const renderBarChart = () => (
    <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis yAxisId="left" orientation="left" tickFormatter={(value) => `${value.toFixed(1)} GOPS`} />
      {showSpeedup && (
        <YAxis yAxisId="right" orientation="right" tickFormatter={(value) => `${value.toFixed(1)}x`} />
      )}
      <Tooltip />
      <Legend />
      <Bar
        yAxisId="left"
        dataKey="throughput"
        fill="#8884d8"
        name="Throughput (GOPS)"
        radius={[2, 2, 0, 0]}
      />
      {showSpeedup && (
        <Bar
          yAxisId="right"
          dataKey="speedup"
          fill="#82ca9d"
          name="Speedup vs FP32"
          radius={[2, 2, 0, 0]}
        />
      )}
    </BarChart>
  );

  const renderRadarChart = () => {
    // Normalize data for radar chart (scale 0-100)
    const maxThroughput = Math.max(...chartData.map(d => d.throughput || 0));
    const maxSpeedup = Math.max(...chartData.map(d => d.speedup || 0));

    const radarData = chartData.map(item => ({
      precision: item.name,
      throughput: maxThroughput > 0 ? (item.throughput / maxThroughput) * 100 : 0,
      speedup: maxSpeedup > 0 ? (item.speedup / maxSpeedup) * 100 : 0
    }));

    return (
      <RadarChart data={radarData} margin={{ top: 20, right: 80, bottom: 20, left: 80 }}>
        <PolarGrid />
        <PolarAngleAxis dataKey="precision" />
        <PolarRadiusAxis
          domain={[0, 100]}
          tick={false}
          tickFormatter={() => ''}
        />
        <Radar
          name="Relative Throughput"
          dataKey="throughput"
          stroke="#8884d8"
          fill="#8884d8"
          fillOpacity={0.3}
          strokeWidth={2}
        />
        {showSpeedup && (
          <Radar
            name="Relative Speedup"
            dataKey="speedup"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.3}
            strokeWidth={2}
          />
        )}
        <Legend />
      </RadarChart>
    );
  };

  // Calculate efficiency scores
  const calculateEfficiency = () => {
    return chartData.map(item => {
      const efficiency = item.speedup || 1; // If no speedup data, assume 1x
      return {
        precision: item.name,
        efficiency: efficiency,
        throughput: item.throughput || 0
      };
    }).sort((a, b) => b.efficiency - a.efficiency);
  };

  const efficiencyRanking = calculateEfficiency();

  return (
    <div className="chart-container">
      <h3>{title}</h3>
      <div className="chart-wrapper" style={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          {chartType === 'radar' ? renderRadarChart() : renderBarChart()}
        </ResponsiveContainer>
      </div>

      <div className="chart-info">
        <p className="chart-description">
          Performance comparison across different numerical precisions.
          {showSpeedup && ' Speedup values show relative performance compared to FP32 baseline.'}
        </p>

        <div className="precision-ranking">
          <h4>Efficiency Ranking:</h4>
          <div className="ranking-list">
            {efficiencyRanking.map((item, index) => (
              <div key={item.precision} className="ranking-item">
                <span className="rank">#{index + 1}</span>
                <span className="precision" style={{ color: getColor(item.precision) }}>
                  {item.precision}
                </span>
                <span className="efficiency">
                  {item.efficiency.toFixed(2)}x speedup
                </span>
                <span className="throughput">
                  {item.throughput.toFixed(1)} GOPS
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-stats">
          <span className="stat-item">
            Best Precision: {efficiencyRanking[0]?.precision || 'N/A'}
          </span>
          <span className="stat-item">
            Max Speedup: {Math.max(...efficiencyRanking.map(r => r.efficiency)).toFixed(2)}x
          </span>
          <span className="stat-item">
            Precisions Tested: {chartData.length}
          </span>
        </div>
      </div>
    </div>
  );
};

export default PrecisionChart;