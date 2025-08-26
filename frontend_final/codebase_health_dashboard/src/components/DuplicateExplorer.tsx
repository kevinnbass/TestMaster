import React, { useState, useMemo } from "react";
import VirtualizedList from "./VirtualizedList";
import { exportDuplicatesToCsv, exportToJson } from "../utils/export";

interface DuplicateExplorerProps {
  groups: string[][];
}

export default function DuplicateExplorer({ groups }: DuplicateExplorerProps) {
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set());
  const [sortBy, setSortBy] = useState<"size" | "files">("size");
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [showAllFiles, setShowAllFiles] = useState<Record<number, boolean>>({});

  const toggleGroup = (index: number) => {
    const newExpanded = new Set(expandedGroups);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedGroups(newExpanded);
  };

  const filteredAndSortedGroups = useMemo(() => {
    let filtered = groups;
    
    if (searchTerm) {
      filtered = groups.filter(group => 
        group.some(file => 
          file.toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    const sorted = [...filtered].sort((a, b) => {
      if (sortBy === "files") {
        return b.length - a.length;
      }
      return b.length - a.length;
    });

    return sorted;
  }, [groups, searchTerm, sortBy]);

  const totalDuplicates = groups.reduce((sum, group) => sum + group.length, 0);
  const totalGroups = groups.length;
  const totalWastedFiles = groups.reduce((sum, group) => sum + Math.max(0, group.length - 1), 0);
  
  const paginatedGroups = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return filteredAndSortedGroups.slice(startIndex, endIndex);
  }, [filteredAndSortedGroups, currentPage, itemsPerPage]);
  
  const totalPages = Math.ceil(filteredAndSortedGroups.length / itemsPerPage);
  
  const toggleShowAllFiles = (groupIndex: number) => {
    setShowAllFiles(prev => ({ ...prev, [groupIndex]: !prev[groupIndex] }));
  };

  const getFileExtension = (filepath: string): string => {
    const parts = filepath.split('.');
    return parts.length > 1 ? parts[parts.length - 1] : 'no-ext';
  };

  const getGroupSummary = (group: string[]): string => {
    const extensions = new Set(group.map(getFileExtension));
    if (extensions.size === 1) {
      return `${Array.from(extensions)[0]} files`;
    }
    return `mixed types`;
  };
  
  const getEstimatedFileSize = (filepath: string): string => {
    // This is a placeholder - in a real app you'd get actual file sizes
    const ext = getFileExtension(filepath);
    const estimates: Record<string, string> = {
      'js': '15KB', 'ts': '12KB', 'py': '8KB', 'json': '5KB', 'md': '3KB'
    };
    return estimates[ext] || '10KB';
  };

  return (
    <div className="duplicate-explorer">
      <div className="duplicate-header">
        <div>
          <h2>Duplicate File Analysis</h2>
          <div className="stats">
            <span className="stat-item">
              <strong>{totalGroups}</strong> duplicate groups
            </span>
            <span className="stat-item">
              <strong>{totalDuplicates}</strong> total files
            </span>
            <span className="stat-item warning">
              <strong>{totalWastedFiles}</strong> redundant files
            </span>
          </div>
        </div>
        
        <div className="controls">
          <input
            type="text"
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value as "size" | "files")}
            className="sort-select"
          >
            <option value="size">Sort by group size</option>
            <option value="files">Sort by file count</option>
          </select>
          
          <select 
            value={itemsPerPage} 
            onChange={(e) => {
              setItemsPerPage(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="items-per-page-select"
          >
            <option value="5">5 per page</option>
            <option value="10">10 per page</option>
            <option value="25">25 per page</option>
            <option value="50">50 per page</option>
          </select>
          
          <div className="export-buttons">
            <button 
              onClick={() => exportDuplicatesToCsv(groups)}
              className="btn btn-small"
              title="Export to CSV"
            >
              📊 CSV
            </button>
            <button 
              onClick={() => exportToJson(groups, { filename: 'duplicates' })}
              className="btn btn-small"
              title="Export to JSON"
            >
              📄 JSON
            </button>
          </div>
        </div>
      </div>

      {filteredAndSortedGroups.length === 0 ? (
        <div className="no-duplicates">
          {searchTerm ? "No duplicates match your search." : "No duplicate files detected."}
        </div>
      ) : (
        <>
          <div className="pagination-info">
            <span>Showing {((currentPage - 1) * itemsPerPage) + 1}-{Math.min(currentPage * itemsPerPage, filteredAndSortedGroups.length)} of {filteredAndSortedGroups.length} groups</span>
          </div>
          
          <div className="duplicate-groups">
            {paginatedGroups.map((g, paginatedIndex) => {
              const originalIndex = groups.indexOf(g);
              const isExpanded = expandedGroups.has(originalIndex);
              const groupSummary = getGroupSummary(g);
              
              return (
              <div key={originalIndex} className="duplicate-group">
                <div 
                  className="group-header"
                  onClick={() => toggleGroup(originalIndex)}
                >
                  <div className="group-info">
                    <span className="group-name">Group {originalIndex + 1}</span>
                    <span className="group-meta">
                      {g.length} files • {groupSummary}
                    </span>
                  </div>
                  <span className="expand-icon">
                    {isExpanded ? "▼" : "▶"}
                  </span>
                </div>
                
                {isExpanded && (
                  <div className="file-list">
                    <div className="file-list-controls">
                      <span className="file-count-info">Showing {showAllFiles[originalIndex] ? g.length : Math.min(g.length, 10)} of {g.length} files</span>
                      {g.length > 10 && (
                        <button 
                          className="toggle-files-btn"
                          onClick={(e) => { e.stopPropagation(); toggleShowAllFiles(originalIndex); }}
                        >
                          {showAllFiles[originalIndex] ? 'Show Less' : 'Show All Files'}
                        </button>
                      )}
                    </div>
                    
                    <div className="files-container">
                      {g.slice(0, showAllFiles[originalIndex] ? g.length : 10).map((p, fileIndex) => {
                        const extension = getFileExtension(p);
                        return (
                          <div key={p} className="file-item">
                            <div className="file-main-info">
                              <span className={`file-extension ext-${extension}`}>
                                {extension}
                              </span>
                              <code className="file-path" title={p}>{p}</code>
                              <div className="file-actions">
                                {fileIndex === 0 ? (
                                  <span className="primary-badge">Keep</span>
                                ) : (
                                  <span className="duplicate-badge">Duplicate</span>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    
                    {g.length > 10 && !showAllFiles[originalIndex] && (
                      <div className="more-files-summary">
                        <span>+ {g.length - 10} more files</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          </div>
          
          {totalPages > 1 && (
            <div className="pagination">
              <button 
                className="pagination-btn"
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
              >
                ← Previous
              </button>
              
              <div className="pagination-pages">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum;
                  if (totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (currentPage <= 3) {
                    pageNum = i + 1;
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i;
                  } else {
                    pageNum = currentPage - 2 + i;
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      className={`pagination-page ${currentPage === pageNum ? 'active' : ''}`}
                      onClick={() => setCurrentPage(pageNum)}
                    >
                      {pageNum}
                    </button>
                  );
                })}
              </div>
              
              <button 
                className="pagination-btn"
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
      
      {totalGroups > 0 && (
        <div className="cleanup-actions">
          <h3>💡 Cleanup Recommendations</h3>
          <div className="cleanup-stats">
            <div className="cleanup-stat">
              <strong>{totalWastedFiles}</strong>
              <span>files can be safely removed</span>
            </div>
            <div className="cleanup-stat">
              <strong>≈{Math.round(totalWastedFiles * 50 / 1024)}KB</strong>
              <span>estimated space savings</span>
            </div>
          </div>
          
          <div className="cleanup-commands">
            <p>Use the generated cleanup scripts to remove duplicate files safely:</p>
            <div className="command-list">
              <div className="command-item">
                <code>scripts\repo_hygiene\dedupe_cleanup.ps1 -WhatIf</code>
                <span className="command-desc">Preview changes (safe)</span>
              </div>
              <div className="command-item">
                <code>scripts\repo_hygiene\dedupe_cleanup.ps1</code>
                <span className="command-desc">Apply cleanup (destructive)</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}