import { useState, useCallback } from 'react';

export interface ColumnDefinition<T> {
  key: keyof T;
  label: string;
  render?: (value: any, item: T) => React.ReactNode;
  sortable?: boolean;
}

export function useColumnReorder<T>(initialColumns: ColumnDefinition<T>[]) {
  const [columns, setColumns] = useState<ColumnDefinition<T>[]>(initialColumns);
  const [draggedColumn, setDraggedColumn] = useState<number | null>(null);

  const handleDragStart = useCallback((index: number) => {
    setDraggedColumn(index);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback((targetIndex: number) => {
    if (draggedColumn === null || draggedColumn === targetIndex) {
      setDraggedColumn(null);
      return;
    }

    const newColumns = [...columns];
    const draggedItem = newColumns[draggedColumn];
    
    // Remove dragged item
    newColumns.splice(draggedColumn, 1);
    
    // Insert at new position
    newColumns.splice(targetIndex, 0, draggedItem);
    
    setColumns(newColumns);
    setDraggedColumn(null);
  }, [columns, draggedColumn]);

  const handleDragEnd = useCallback(() => {
    setDraggedColumn(null);
  }, []);

  return {
    columns,
    draggedColumn,
    handleDragStart,
    handleDragOver,
    handleDrop,
    handleDragEnd
  };
}