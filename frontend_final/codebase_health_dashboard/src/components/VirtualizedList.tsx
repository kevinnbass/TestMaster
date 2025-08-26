import React, { useState, useMemo } from 'react';

interface VirtualizedListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemHeight?: number;
  maxHeight?: number;
  pageSize?: number;
  searchable?: boolean;
  searchKey?: keyof T;
  emptyMessage?: string;
}

export default function VirtualizedList<T>({
  items,
  renderItem,
  itemHeight = 40,
  maxHeight = 400,
  pageSize = 50,
  searchable = false,
  searchKey,
  emptyMessage = 'No items found'
}: VirtualizedListProps<T>) {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(0);

  const filteredItems = useMemo(() => {
    if (!searchable || !searchTerm || !searchKey) return items;
    
    const term = searchTerm.toLowerCase();
    return items.filter(item => {
      const value = item[searchKey];
      return String(value).toLowerCase().includes(term);
    });
  }, [items, searchTerm, searchKey, searchable]);

  const totalPages = Math.ceil(filteredItems.length / pageSize);
  const paginatedItems = useMemo(() => {
    const start = currentPage * pageSize;
    const end = start + pageSize;
    return filteredItems.slice(start, end);
  }, [filteredItems, currentPage, pageSize]);

  const goToPage = (page: number) => {
    setCurrentPage(Math.max(0, Math.min(page, totalPages - 1)));
  };

  if (filteredItems.length === 0) {
    return (
      <div className="virtualized-list">
        {searchable && (
          <div className="search-bar">
            <input
              type="text"
              placeholder={`Search ${searchKey ? String(searchKey) : 'items'}...`}
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(0);
              }}
              className="search-input"
            />
          </div>
        )}
        <div className="empty-state">{emptyMessage}</div>
      </div>
    );
  }

  return (
    <div className="virtualized-list">
      {searchable && (
        <div className="search-bar">
          <input
            type="text"
            placeholder={`Search ${searchKey ? String(searchKey) : 'items'}...`}
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(0);
            }}
            className="search-input"
          />
          <span className="search-count">
            {filteredItems.length} of {items.length} items
          </span>
        </div>
      )}

      <div 
        className="list-container"
        style={{ maxHeight, overflowY: 'auto' }}
      >
        {paginatedItems.map((item, index) => (
          <div key={currentPage * pageSize + index} style={{ height: itemHeight }}>
            {renderItem(item, currentPage * pageSize + index)}
          </div>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage === 0}
            className="pagination-btn"
          >
            Previous
          </button>
          <span className="pagination-info">
            Page {currentPage + 1} of {totalPages} 
            ({filteredItems.length} items)
          </span>
          <button
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage === totalPages - 1}
            className="pagination-btn"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}