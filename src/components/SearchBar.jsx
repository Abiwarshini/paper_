import React, { useState } from 'react';
import { Search } from 'lucide-react';
import '../css/SearchBar.css';

const SearchBar = ({ onSearch, placeholder = 'Search children by name...' }) => {
  const [term, setTerm] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(term);
  };

  const handleClear = () => {
    setTerm('');
    onSearch('');
  };

  return (
    <form className="search-bar-form" onSubmit={handleSubmit}>
      <div className="search-input-wrapper">
        <Search className="search-bar-icon" size={18} />
        <input
          type="text"
          className="search-bar-input"
          placeholder={placeholder}
          value={term}
          onChange={(e) => setTerm(e.target.value)}
        />
        {term && (
          <button type="button" className="search-clear-btn" onClick={handleClear}>
            Clear
          </button>
        )}
      </div>
      <button type="submit" className="search-bar-btn">
        Search
      </button>
    </form>
  );
};

export default SearchBar;
