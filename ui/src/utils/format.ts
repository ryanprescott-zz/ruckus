/**
 * Utility functions for formatting data in the UI
 */

/**
 * Format uptime seconds into a human-readable string
 */
export function formatUptime(seconds: number): string {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`;
  }
  
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${Math.floor(seconds % 60)}s`;
  }
  
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ${Math.floor(minutes % 60)}m ${Math.floor(seconds % 60)}s`;
  }
  
  const days = Math.floor(hours / 24);
  return `${days}d ${Math.floor(hours % 24)}h ${Math.floor(minutes % 60)}m ${Math.floor(seconds % 60)}s`;
}

/**
 * Format ISO timestamp into a readable date/time string
 */
export function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString('en-US', {
    month: '2-digit',
    day: '2-digit', 
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
}

/**
 * Format agent details for display in text areas with complete nested content
 */
export function formatAgentDetails(
  _title: string,
  data: Record<string, any>
): string {
  if (!data || Object.keys(data).length === 0) {
    return `No data available`;
  }

  try {
    // Separate error fields for special handling
    const errorFields = ['error', 'error_type', 'traceback'];
    const errorEntries: string[] = [];
    const regularEntries: string[] = [];

    Object.entries(data).forEach(([key, value]) => {
      const fieldName = formatFieldName(key);
      const fieldValue = formatFieldValue(value, '');

      let formattedEntry: string;
      if (fieldValue.includes('\n')) {
        formattedEntry = `${fieldName}:\n  ${fieldValue.replace(/\n/g, '\n  ')}`;
      } else {
        formattedEntry = `${fieldName}: ${fieldValue}`;
      }

      // Highlight error fields
      if (errorFields.includes(key) && value !== null && value !== undefined && value !== '') {
        if (key === 'error') {
          errorEntries.push(`🚨 ${formattedEntry}`);
        } else if (key === 'error_type') {
          errorEntries.push(`⚠️  ${formattedEntry}`);
        } else if (key === 'traceback') {
          errorEntries.push(`📋 ${formattedEntry}`);
        } else {
          errorEntries.push(`❌ ${formattedEntry}`);
        }
      } else {
        regularEntries.push(formattedEntry);
      }
    });

    // Show error information first if present
    const allEntries = [...errorEntries, ...regularEntries];
    return allEntries.join('\n\n');
  } catch (error) {
    return `Error formatting data: ${error}`;
  }
}

/**
 * Format field names to be more readable
 */
function formatFieldName(key: string): string {
  return key
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

/**
 * Format field values for display - shows complete content recursively
 */
function formatFieldValue(value: any, indent: string = ''): string {
  if (value === null || value === undefined) {
    return 'N/A';
  }
  
  if (typeof value === 'object') {
    if (Array.isArray(value)) {
      if (value.length === 0) {
        return 'None';
      }
      
      // For arrays of primitives, join with commas
      if (value.every(item => typeof item !== 'object')) {
        return value.join(', ');
      }
      
      // For arrays of objects, format each item on its own line
      return value.map((item, index) => {
        const itemValue = formatFieldValue(item, indent + '  ');
        return `${indent}[${index}]: ${itemValue}`;
      }).join('\n');
    }
    
    // For objects, show all key-value pairs
    const keys = Object.keys(value);
    if (keys.length === 0) {
      return 'Empty';
    }
    
    // Format all keys and values with proper indentation
    const entries = keys.map(key => {
      const fieldName = formatFieldName(key);
      const fieldValue = formatFieldValue(value[key], indent + '  ');
      
      // If the value contains newlines, format it properly
      if (fieldValue.includes('\n')) {
        return `${indent}${fieldName}:\n${indent}  ${fieldValue.replace(/\n/g, '\n' + indent + '  ')}`;
      }
      
      return `${indent}${fieldName}: ${fieldValue}`;
    });
    
    return entries.join('\n');
  }
  
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  
  // Handle timestamps
  if (typeof value === 'string' && value.match(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)) {
    return formatTimestamp(value);
  }
  
  return String(value);
}