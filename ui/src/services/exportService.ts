import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import type { ExperimentResult } from '../types/api';
import type { ProcessedChartData } from '../utils/chartDataProcessor';

export interface ExportOptions {
  format: 'png' | 'svg' | 'csv' | 'pdf' | 'json';
  quality?: number; // For PNG (0-1)
  includeMetadata?: boolean;
  filename?: string;
  charts?: string[]; // Which charts to include
}

export class ExportService {
  /**
   * Export a chart as PNG image
   */
  static async exportChartAsPNG(
    chartElement: HTMLElement,
    options: { filename?: string; quality?: number } = {}
  ): Promise<void> {
    try {
      const canvas = await html2canvas(chartElement, {
        useCORS: true,
        scale: 2, // Higher resolution
        backgroundColor: '#ffffff',
        removeContainer: true
      });

      // Convert canvas to blob
      const quality = options.quality || 0.95;
      canvas.toBlob(
        (blob) => {
          if (blob) {
            this.downloadBlob(blob, options.filename || 'chart.png');
          }
        },
        'image/png',
        quality
      );
    } catch (error) {
      console.error('Failed to export chart as PNG:', error);
      throw new Error('PNG export failed');
    }
  }

  /**
   * Export chart data as CSV
   */
  static exportChartDataAsCSV(
    data: any[],
    options: { filename?: string } = {}
  ): void {
    if (!data || data.length === 0) {
      throw new Error('No data available to export');
    }

    // Get all unique keys from the data
    const keys = Array.from(
      new Set(data.flatMap(item => Object.keys(item)))
    );

    // Create CSV content
    const csvContent = [
      // Header row
      keys.join(','),
      // Data rows
      ...data.map(item =>
        keys.map(key => {
          const value = item[key];
          // Handle values that might contain commas
          if (typeof value === 'string' && value.includes(',')) {
            return `"${value}"`;
          }
          return value || '';
        }).join(',')
      )
    ].join('\n');

    // Create and download blob
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    this.downloadBlob(blob, options.filename || 'chart-data.csv');
  }

  /**
   * Export experiment result as JSON
   */
  static exportAsJSON(
    result: ExperimentResult,
    options: { filename?: string; formatted?: boolean } = {}
  ): void {
    const jsonContent = JSON.stringify(
      result,
      null,
      options.formatted !== false ? 2 : 0
    );

    const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
    this.downloadBlob(
      blob,
      options.filename || `experiment-${result.experiment_id}-${result.job_id}.json`
    );
  }

  /**
   * Generate comprehensive PDF report
   */
  static async generatePDFReport(
    experimentResult: ExperimentResult,
    processedData: ProcessedChartData,
    chartElements: { [chartType: string]: HTMLElement },
    options: { filename?: string } = {}
  ): Promise<void> {
    try {
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 20;
      let yPosition = margin;

      // Title page
      pdf.setFontSize(20);
      pdf.text('Benchmark Report', margin, yPosition);
      yPosition += 15;

      pdf.setFontSize(12);
      pdf.text(`Experiment: ${experimentResult.experiment_id}`, margin, yPosition);
      yPosition += 7;
      pdf.text(`Job: ${experimentResult.job_id}`, margin, yPosition);
      yPosition += 7;
      pdf.text(`Agent: ${experimentResult.agent_id}`, margin, yPosition);
      yPosition += 7;
      pdf.text(`Type: ${processedData.type}`, margin, yPosition);
      yPosition += 7;
      pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, yPosition);
      yPosition += 15;

      // Add summary statistics
      if (Object.keys(processedData.charts).length > 0) {
        pdf.setFontSize(14);
        pdf.text('Summary', margin, yPosition);
        yPosition += 10;

        pdf.setFontSize(10);
        Object.entries(processedData.charts).forEach(([chartType, data]) => {
          if (data && data.length > 0) {
            pdf.text(`${chartType}: ${data.length} data points`, margin, yPosition);
            yPosition += 5;
          }
        });
        yPosition += 10;
      }

      // Add charts
      let chartIndex = 0;
      for (const [chartType, element] of Object.entries(chartElements)) {
        if (element) {
          // Start new page for each chart (except first)
          if (chartIndex > 0) {
            pdf.addPage();
            yPosition = margin;
          }

          // Chart title
          pdf.setFontSize(14);
          pdf.text(this.getChartTitle(chartType), margin, yPosition);
          yPosition += 15;

          try {
            // Capture chart as image
            const canvas = await html2canvas(element, {
              useCORS: true,
              scale: 1,
              backgroundColor: '#ffffff'
            });

            // Add image to PDF
            const imgData = canvas.toDataURL('image/png');
            const imgWidth = pageWidth - (margin * 2);
            const imgHeight = (canvas.height * imgWidth) / canvas.width;

            pdf.addImage(imgData, 'PNG', margin, yPosition, imgWidth, imgHeight);
            yPosition += imgHeight + 10;
          } catch (error) {
            console.error(`Failed to capture chart ${chartType}:`, error);
            pdf.text(`[Chart could not be rendered: ${chartType}]`, margin, yPosition);
            yPosition += 10;
          }

          chartIndex++;
        }
      }

      // Add raw data page
      pdf.addPage();
      yPosition = margin;

      pdf.setFontSize(14);
      pdf.text('Raw Data', margin, yPosition);
      yPosition += 10;

      pdf.setFontSize(8);
      const rawDataText = JSON.stringify(experimentResult.output, null, 2);
      const lines = pdf.splitTextToSize(rawDataText, pageWidth - (margin * 2));

      lines.forEach((line: string) => {
        if (yPosition > pageHeight - margin) {
          pdf.addPage();
          yPosition = margin;
        }
        pdf.text(line, margin, yPosition);
        yPosition += 3;
      });

      // Save PDF
      const filename = options.filename || `benchmark-report-${experimentResult.job_id}.pdf`;
      pdf.save(filename);
    } catch (error) {
      console.error('Failed to generate PDF report:', error);
      throw new Error('PDF generation failed');
    }
  }

  /**
   * Generate Markdown report
   */
  static generateMarkdownReport(
    experimentResult: ExperimentResult,
    processedData: ProcessedChartData
  ): string {
    const lines: string[] = [];

    // Header
    lines.push(`# Benchmark Report`);
    lines.push('');
    lines.push(`**Experiment ID:** ${experimentResult.experiment_id}`);
    lines.push(`**Job ID:** ${experimentResult.job_id}`);
    lines.push(`**Agent ID:** ${experimentResult.agent_id}`);
    lines.push(`**Benchmark Type:** ${processedData.type}`);
    if (processedData.metadata.device) {
      lines.push(`**Device:** ${processedData.metadata.device}`);
    }
    lines.push(`**Generated:** ${new Date().toISOString()}`);
    lines.push('');

    // Summary
    lines.push('## Summary');
    lines.push('');
    Object.entries(processedData.charts).forEach(([chartType, data]) => {
      if (data && data.length > 0) {
        lines.push(`- **${chartType}:** ${data.length} data points`);
      }
    });
    lines.push('');

    // Chart data
    Object.entries(processedData.charts).forEach(([chartType, data]) => {
      if (data && data.length > 0) {
        lines.push(`## ${this.getChartTitle(chartType)}`);
        lines.push('');

        // Create table
        lines.push('| Metric | Value | Unit |');
        lines.push('|--------|-------|------|');

        data.forEach(point => {
          lines.push(`| ${point.name} | ${point.value.toFixed(3)} | ${point.unit} |`);
        });

        lines.push('');
      }
    });

    // Raw data
    lines.push('## Raw Data');
    lines.push('');
    lines.push('```json');
    lines.push(JSON.stringify(experimentResult.output, null, 2));
    lines.push('```');

    return lines.join('\n');
  }

  /**
   * Create shareable URL (placeholder - would need backend support)
   */
  static generateShareableURL(experimentResult: ExperimentResult): string {
    // This would typically involve uploading data to a temporary storage
    // and returning a shareable URL. For now, we'll return a placeholder.
    const baseUrl = window.location.origin;
    const shareId = `${experimentResult.job_id}-${Date.now()}`;
    return `${baseUrl}/shared/${shareId}`;
  }

  /**
   * Download blob as file
   */
  private static downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /**
   * Get human-readable chart title
   */
  private static getChartTitle(chartType: string): string {
    switch (chartType) {
      case 'memory-bandwidth': return 'Memory Bandwidth Performance';
      case 'compute-flops': return 'Compute Performance (FLOPS)';
      case 'precision-comparison': return 'Precision Performance Comparison';
      case 'llm-performance': return 'LLM Performance Metrics';
      default: return chartType.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
  }

  /**
   * Generate filename with timestamp
   */
  static generateFilename(
    prefix: string,
    experimentId: string,
    extension: string
  ): string {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
    return `${prefix}-${experimentId}-${timestamp}.${extension}`;
  }
}