import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import '../css/Pagination.css';

const Pagination = ({ currentPage, totalPages, totalRecords, limit, onPageChange }) => {
  if (totalPages <= 1) return null;

  const getPageNumbers = () => {
    const pages = [];
    for (let i = 1; i <= totalPages; i++) {
      pages.push(i);
    }
    return pages;
  };

  return (
    <div className="pagination-container">
      <div className="pagination-info">
        Showing Page <span className="highlight">{currentPage}</span> of{' '}
        <span className="highlight">{totalPages}</span> ({totalRecords} records)
      </div>
      <div className="pagination-controls">
        <button
          className="pagination-btn prev"
          disabled={currentPage === 1}
          onClick={() => onPageChange(currentPage - 1)}
        >
          <ChevronLeft size={16} />
          <span>Previous</span>
        </button>

        {getPageNumbers().map((num) => (
          <button
            key={num}
            className={`pagination-number ${currentPage === num ? 'active' : ''}`}
            onClick={() => onPageChange(num)}
          >
            {num}
          </button>
        ))}

        <button
          className="pagination-btn next"
          disabled={currentPage === totalPages}
          onClick={() => onPageChange(currentPage + 1)}
        >
          <span>Next</span>
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
};

export default Pagination;
