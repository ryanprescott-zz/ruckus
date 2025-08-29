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
 * Format agent details for display in text areas with simple key-value format
 */
export function formatAgentDetails(
  title: string, 
  data: Record<string, any>
): string {
  if (!data || Object.keys(data).length === 0) {
    return `No data available`;
  }
  
  try {
    // Create simple key-value format
    const entries = Object.entries(data);
    
    // Format each entry as simple key-value pairs
    const formattedEntries = entries.map(([key, value]) => {
      const fieldName = formatFieldName(key);
      const fieldValue = formatFieldValue(value);
      return `${fieldName}: ${fieldValue}`;
    });
    
    return formattedEntries.join('\n');
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
 * Format field values for display
 */
function formatFieldValue(value: any): string {
  if (value === null || value === undefined) {
    return 'N/A';
  }
  
  if (typeof value === 'object') {
    if (Array.isArray(value)) {
      return value.length > 0 ? value.join(', ') : 'None';
    }
    
    // For nested objects, show a summary or key info
    const keys = Object.keys(value);
    if (keys.length === 0) {
      return 'Empty';
    }
    
    // For small objects, show key-value pairs inline
    if (keys.length <= 3) {
      return keys.map(k => `${k}: ${value[k]}`).join(', ');
    }
    
    // For larger objects, show count
    return `${keys.length} items: ${keys.slice(0, 2).join(', ')}...`;
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