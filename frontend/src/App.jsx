import React, { useState } from 'react';
import axios from 'axios';
import { Search, FileText, Image as ImageIcon, Send, Loader2 } from 'lucide-react';

// NOTE: In a real SharePoint webpart, we would use the generic SPHttpClient. 
// For this demo, we call our local FastAPI backend.

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/api/ask', {
        question: query
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError("Failed to retrieve information. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* SharePoint-style Header */}
      <header className="bg-sharepoint-header text-white h-12 flex items-center px-4 shadow-sm">
        <div className="flex items-center gap-2 font-semibold text-lg">
          <div className="grid grid-cols-2 gap-0.5 w-6 h-6 bg-white/10 p-1 rounded">
            <div className="bg-white rounded-[1px]"></div>
            <div className="bg-white rounded-[1px]"></div>
            <div className="bg-white rounded-[1px]"></div>
            <div className="bg-white rounded-[1px]"></div>
          </div>
          <span>Sharjah HR Assistant</span>
        </div>
        <div className="ml-auto text-sm opacity-90">Demo Webpart</div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-6">

        {/* Search Section */}
        <div className="bg-white rounded-lg shadow-sm p-8 mb-6 border border-gray-200">
          <h1 className="text-2xl font-bold text-gray-800 mb-2">How can I help you today?</h1>
          <p className="text-gray-500 mb-6">Ask about regulations, allowances, or specialized contracts.</p>

          <form onSubmit={handleSearch} className="relative max-w-3xl">
            <input
              type="text"
              className="w-full pl-12 pr-4 py-3 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-sharepoint-blue focus:border-transparent transition-all shadow-sm"
              placeholder="e.g., What is the technical allowance for a Consultant Doctor?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <Search className="absolute left-4 top-3.5 text-gray-400 w-5 h-5" />
            <button
              type="submit"
              disabled={loading}
              className="absolute right-2 top-2 bg-sharepoint-blue text-white p-1.5 rounded-full hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-6 border border-red-200">
            {error}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Left: LLM Answer */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
              <div className="bg-gray-50 px-4 py-3 border-b border-gray-100 flex items-center gap-2">
                <FileText className="w-4 h-4 text-sharepoint-blue" />
                <h2 className="font-semibold text-gray-700">Answer</h2>
              </div>
              <div className="p-6">
                <div className="prose prose-blue max-w-none text-gray-800 whitespace-pre-wrap leading-relaxed">
                  {result.answer}
                </div>
                {result.context_caption && (
                  <div className="mt-6 pt-4 border-t border-gray-100 text-xs text-gray-500">
                    Source: {result.context_caption}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Source Table Image */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden flex flex-col">
              <div className="bg-gray-50 px-4 py-3 border-b border-gray-100 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-sharepoint-blue" />
                <h2 className="font-semibold text-gray-700">Reference Table</h2>
              </div>

              <div className="flex-1 bg-gray-100 p-4 flex items-center justify-center min-h-[300px]">
                {result.image_paths && result.image_paths.length > 0 ? (
                  /* 
                     NOTE: We need to serve these images. 
                     For this demo, we can't easily serve local C:\ paths nicely in browser without a static file server.
                     However, since this is a local demo, we might encounter a 'Not allowed to load local resource' error.
                     
                     FIX: We will update the backend API to serve these images via an endpoint later. 
                     For now, let's assume the API will return a URL or we try to display it.
                  */
                  <div className="space-y-4 w-full">
                    {result.image_paths.map((path, idx) => {
                      // Backend now returns full URL (e.g. http://localhost:8000/static/file.png)
                      // We can use it directly.
                      // For filename display, we extract it.
                      const filename = path.split('\\').pop().split('/').pop();

                      return (
                        <div key={idx} className="bg-white p-2 rounded shadow-sm">
                          <img
                            src={path}
                            alt={`Table source ${idx + 1}`}
                            className="w-full h-auto rounded border border-gray-200"
                            onError={(e) => {
                              console.error("Image load error:", path);
                              e.target.onerror = null;
                              // e.target.src = "https://placehold.co/600x400?text=Image+Load+Error"; 
                              // Keep error placeholder if it fails
                            }}
                          />
                          <p className="text-center text-xs text-gray-400 mt-2">{filename}</p>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-gray-400 text-sm">No visual reference available for this section.</div>
                )}
              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}

export default App;
