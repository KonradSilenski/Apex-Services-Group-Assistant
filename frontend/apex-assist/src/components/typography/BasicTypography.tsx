import useSWR from "swr";
import { useState, useEffect } from "react";

const fetcher = (url: string) => {
  const token = localStorage.getItem("token");
  return fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  }).then((res) => {
    if (!res.ok) throw new Error("Network response was not ok");
    return res.json();
  });
};

interface SorRow {
  id: string;
  job_type: string;
  short: string;
  medium: string;
  element: string;
  work_categories: string;
  work_sub_categories: string;
  work_sub_categories_attributes: string;
  counter: number | null;
}

const BasicTypography = () => {

  const [rows, setRows] = useState<SorRow[]>([]);
  const [loading, setLoading] = useState(true);

  const { data: visits } = useSWR(
    `${import.meta.env.VITE_APP_API_BASE_URL}/stats/visits`,
    fetcher
  );

  const { data: codes } = useSWR(
    `${import.meta.env.VITE_APP_API_BASE_URL}/stats/codes`,
    fetcher
  );

  useEffect(() => {
    fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/sor/most_used?limit=5`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem("token") || ""}`,
      },
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Most used API response:", data);

        if (Array.isArray(data)) {
          setRows(data);
        } else {
          console.error("Expected an array but got:", data);
          setRows([]);
        }

        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching most used rows:", error);
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading...</p>;

  return (
    <div className="rounded-xl dark:shadow-dark-md shadow-md bg-white dark:bg-darkgray p-6 relative w-full break-words">
      <h5 className="card-title">Statistics:</h5>
      <div className="mt-6">
        <div className="border border-ld rounded-xl px-6 py-4 mb-6">
          <p className="font-semibold text-2xl text-black">Visits completed using the Assistant: {visits}</p>
        </div>
        <div className="border border-ld rounded-xl px-6 py-4 mb-6">
          <p className="font-semibold text-2xl text-black">Number of SOR codes generated: {codes}</p>
        </div>
        <div className="border border-ld rounded-xl px-6 py-4 mb-6">
          <p className="font-semibold text-2xl text-black">Most commonly used SOR codes:</p>
        </div>
        <div className="border border-ld rounded-xl px-6 py-4 mb-6">
          <ul className="space-y-2 px-6 py-4">
            {rows.map((row) => (
              <li key={row.id} className="p-2 border rounded-md shadow-sm text-blue-800">
                <p><strong>ID:</strong> {row.id}</p>
                <p><strong>Short:</strong> {row.short || "—"}</p>
                <p><strong>Counter:</strong> {row.counter ?? "—"}</p>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}

export default BasicTypography