import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Badge, Table } from "flowbite-react";
import SimpleBar from "simplebar-react";
import { useNavigate } from "react-router";


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


const ProductRevenue = () => {
  const navigate = useNavigate();
  const { data: jobs, error, isLoading } = useSWR(`${import.meta.env.VITE_APP_API_BASE_URL}/jobs/all`, fetcher, {
    refreshInterval: 5000,
  });

  //const [newJobNotice, setNewJobNotice] = useState(null);
  const [sortKey, setSortKey] = useState("id");
  const [sortDirection, setSortDirection] = useState("asc");

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDirection("asc");
    }
  };

  const sortedJobs = useMemo(() => {
    if (!jobs) return [];
    return [...jobs].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === "string") {
        return sortDirection === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }
      return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
    });
  }, [jobs, sortKey, sortDirection]);

  useEffect(() => {
    if (!jobs) return;

    //const seenJobIds = JSON.parse(localStorage.getItem("knownJobIds") || "[]");
    const currentJobIds = jobs.map((job: any) => job.date);
    //const newJobs = jobs.filter((job: any) => !seenJobIds.includes(job.id));

    /*
    if (newJobs.length > 0) {
      setNewJobNotice(newJobs[0]);
      setTimeout(() => setNewJobNotice(null), 4000);
    }
    */

    localStorage.setItem("knownJobIds", JSON.stringify(currentJobIds));
  }, [jobs]);

  if (isLoading) return <div>Loading jobs...</div>;
  if (error) return <div>Failed to load jobs.</div>;

  return (
    <div className="rounded-xl dark:shadow-dark-md shadow-md bg-white dark:bg-darkgray p-6  relative w-full break-words">
      <div className="px-6">
        <h5 className="card-title mb-6">To Be Reviewed</h5>
      </div>
      <SimpleBar className="max-h-[450px]">
        <div className="overflow-x-auto">
          <Table hoverable>
            <Table.Head>
              <Table.HeadCell className="p-6" onClick={() => handleSort("client")}>
                Details
              </Table.HeadCell>
              <Table.HeadCell onClick={() => handleSort("property")}>Property</Table.HeadCell>
              <Table.HeadCell onClick={() => handleSort("date")}>Date</Table.HeadCell>
            </Table.Head>
            <Table.Body className="divide-y divide-border dark:divide-darkborder">
              {sortedJobs.map((item: any, index: number) => (
                <Table.Row
                  key={index}
                  onClick={() => navigate(`/job?jobId=${item.id}`)}
                  className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  <Table.Cell className="whitespace-nowrap ps-6">
                    <div className="flex gap-3 items-center">
                      <div className="truncat line-clamp-2 sm:text-wrap max-w-56">
                        <h6 className="text-sm">{item.client}</h6>
                        <p className="text-xs ">{item.propeller_id}</p>
                      </div>
                    </div>
                  </Table.Cell>
                  <Table.Cell>
                    <div className="me-5">
                      <p className="text-base">{item.property}</p>
                    </div>
                  </Table.Cell>
                  <Table.Cell>
                    <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300">
                      {item.date}
                    </Badge>
                  </Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>

          </Table>
        </div>
      </SimpleBar>
    </div>
  );
};

export default ProductRevenue;
