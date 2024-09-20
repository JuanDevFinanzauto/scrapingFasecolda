import React, { useState } from 'react';
import axios from 'axios';

interface FileResponse {
  status: string;
  files_moved: string[];
  message?: string;
}

const App: React.FC = () => {
  const [files, setFiles] = useState<string[]>([]);
  const [message, setMessage] = useState<string>('');

  const handleScrape = async () => {
    setMessage('Iniciando scraping y descarga de archivos...');
    try {
      const response = await axios.get<FileResponse>('http://localhost:5000/scrape');
      if (response.data.status === 'success') {
        setFiles(response.data.files_moved);
        setMessage('Archivos descargados y descomprimidos con éxito.');
      } else {
        setMessage(response.data.message || 'Error al procesar la solicitud.');
      }
    } catch (error) {
      setMessage('Error al conectar con el servidor.');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto bg-white p-8 shadow-lg rounded-lg">
        <h1 className="text-2xl font-bold text-center mb-4">Sistema de Scraping Empresarial</h1>
        {message && <div className="bg-blue-100 text-blue-800 p-2 rounded mb-4">{message}</div>}
        <button
          onClick={handleScrape}
          className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition mb-4"
        >
          Iniciar Scraping y Descarga
        </button>
        {files.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold mb-2">Archivos Descargados:</h2>
            <ul className="list-disc pl-5">
              {files.map((file, index) => (
                <li key={index}>{file}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;

