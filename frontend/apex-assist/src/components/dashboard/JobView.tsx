import { useSearchParams, useNavigate } from 'react-router';
import SimpleBar from "simplebar-react";
import useSWR, { mutate } from "swr";
import { useState } from "react";
import { useEffect } from "react";
import SOROverlay from "./SOROverlay";
import { loadImagesByJobId } from 'src/utils/loadImages';
import Lightbox from "yet-another-react-lightbox";
import Zoom from "yet-another-react-lightbox/plugins/zoom";
import "yet-another-react-lightbox/styles.css";


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

const JobView = () => {
    const [overlayOpen, setOverlayOpen] = useState<{ open: boolean; type: "completed" | "future" | null }>({
        open: false,
        type: null,
    });

    const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
    const [confirmSubmitOpen, setConfirmSubmitOpen] = useState(false);
    const [selectedClient, setSelectedClient] = useState("");
    const [searchParams] = useSearchParams();
    const jobId = searchParams.get('jobId');
    const [selectedSorId, setSelectedSorId] = useState<string | null>(null);
    const [isExpanded, setIsExpanded] = useState(false);
    const [images, setImages] = useState<string[]>([]);
    const navigate = useNavigate();
    const [quant, setQuant] = useState(0);
    const [lightboxOpen, setLightboxOpen] = useState(false);
    const [lightboxIndex, setLightboxIndex] = useState(0);

    const [setQuantity, setConfirmQuantityOpen] = useState({
        state: false,
        quantity: 0,
        code: ""
    });

    useEffect(() => {
        const fetchImages = async () => {
            if (jobId) {
                const loaded = await loadImagesByJobId(jobId);
                const formatted = loaded.map(base64 => `data:image/png;base64,${base64}`);
                setImages(formatted);
            }
        };
        fetchImages();
    }, [jobId]);




    const changeState = (newState: boolean, code: string) => {
        changeCode(code);
        setConfirmQuantityOpen(previousState => {
            return { ...previousState, state: newState }
        });
    }

    const changeCode = (newCode: string) => {
        setConfirmQuantityOpen(previousState => {
            return { ...previousState, code: newCode }
        });
    }

    const changeQuantity = async (newQuantity: number) => {
        if (!jobId || !selectedSorId) return;
        try {
            const response = await fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/jobs/change_code_quantity?job_id=${jobId}&sor=${selectedSorId}&quantity=${newQuantity}`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                },
            });

            if (!response.ok) throw new Error("Failed to change the quantity");

            setSelectedSorId(null);
            changeState(false, "")
            refreshPage();
        } catch (error) {
            console.error("Change failed:", error);
            alert("Failed to dchange quantity.");
        }

    }

    const { data: job, error, isLoading } = useSWR(
        jobId ? `${import.meta.env.VITE_APP_API_BASE_URL}/jobs/id?job_id=${jobId}` : null,
        fetcher
    );

    const { data: jobDetails } = useSWR(
        jobId ? `${import.meta.env.VITE_APP_API_BASE_URL}/details/?id=${jobId}` : null,
        fetcher
    );

    const refreshPage = () => {
        if (jobId) {
            mutate(`${import.meta.env.VITE_APP_API_BASE_URL}/jobs/id?job_id=${jobId}`);
        }
    };

    if (isLoading) return <div>Loading...</div>;
    if (error) return <div>Error loading job data</div>;
    if (!job) return <div>No job found</div>;
    if (!jobDetails) return <div>No details found.</div>

    const handleSelectSor = (id: string) => {
        if (selectedSorId === id) {
            setSelectedSorId(null);
            setIsExpanded(false);
        } else {
            setSelectedSorId(id);
            setIsExpanded(true);
        }
    };


    const handleDeleteSor = async () => {
        if (!jobId || !selectedSorId) return;

        try {
            const response = await fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/jobs/remove_sor?job_id=${jobId}&sor=${selectedSorId}`, {
                method: "DELETE",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                },
            });

            if (!response.ok) throw new Error("Failed to delete SOR code");

            setConfirmDeleteOpen(false);
            setSelectedSorId(null);
            setIsExpanded(false);
            refreshPage();
        } catch (error) {
            console.error("Delete failed:", error);
            alert("Failed to delete SOR code.");
        }
    };

    const handleSubmitSor = async () => {
        if (!jobId) return;

        try {
            const response = await fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/jobs/submit?job_id=${jobId}&propeller_id=${job.propeller_id}&client=${encodeURIComponent(selectedClient)}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                },
            });

            if (!response.ok) throw new Error("Failed to submit to Propeller.");

            setConfirmSubmitOpen(false);
            navigate("/");
        } catch (error) {
            console.error("Submit failed:", error);
            alert("Failed to submit to Propeller.");
        }
    };

    const selectedSor = job.sor_codes.find((code: any) => code.sor.id === selectedSorId);

    return (
        <div className="rounded-xl dark:shadow-dark-md shadow-md bg-white dark:bg-darkgray p-6 relative w-full break-words">
            <div className="px-6">
                <div className="flex justify-between items-center mb-6">
                    <h5 className="card-title">Visit</h5>
                    <button
                        onClick={() => navigate("/")}
                        className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded shadow"
                    >
                        ←
                    </button>
                </div>

                <SimpleBar className="mb-4">
                    <div className="overflow-x-auto space-y-4 pb-1">
                        <div><strong>Visit ID:</strong> {job.propeller_id}</div>
                        <div><strong>Tenant:</strong> {job.client}</div>
                        <div><strong>Property:</strong> {job.property}</div>

                        <div className="mt-4 mb-4">
                            <div className="mt-4">
                                {(() => {
                                    const renderDetail = (label: string, value
                                        : any) => {
                                        if (value == " ") {
                                            return null;
                                        }

                                        return (
                                            <p>
                                                {label}: {value}
                                            </p>
                                        );
                                    };


                                    return (
                                        <>
                                            {/* General */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>General</h2>
                                                    {renderDetail("Visit Type", jobDetails.visit_type)}
                                                    {renderDetail("Work Description", jobDetails.work_desc)}
                                                    {renderDetail("Property Type", jobDetails.property_type)}
                                                </div>
                                            </SimpleBar>

                                            {/* Scaffolding */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Scaffolding</h2>
                                                    {renderDetail("Scaffold Required", jobDetails.scaffold_required)}
                                                    {renderDetail("Scaffold Type", jobDetails.scaffold_type)}
                                                    {renderDetail("Elevation Measurement", jobDetails.elevation_measurement)}
                                                </div>
                                            </SimpleBar>

                                            {/* Roof */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Roof</h2>
                                                    {renderDetail("Roof Type", jobDetails.roof_type)}
                                                    {renderDetail("Coverings Type", jobDetails.coverings_type)}
                                                    {renderDetail("Tile Size", jobDetails.tile_size)}
                                                    {renderDetail("Roof Measurement", jobDetails.roof_measurement)}
                                                </div>
                                            </SimpleBar>

                                            {/* Ridge Tile */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Ridge Tile</h2>
                                                    {renderDetail("Ridge Tile", jobDetails.ridge_tile)}
                                                    {renderDetail("Ridge Tile Type", jobDetails.ridge_tile_type)}
                                                    {renderDetail("Ridge Job", jobDetails.ridge_job)}
                                                    {renderDetail("Ridge Measurement", jobDetails.ridge_measurement)}
                                                </div>
                                            </SimpleBar>

                                            {/* Leadwork */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Leadwork</h2>
                                                    {renderDetail("Leadwork", jobDetails.leadwork)}
                                                    {renderDetail("Measurement", jobDetails.leadwork_measurement)}
                                                    {renderDetail("Comment", jobDetails.leadwork_comment)}
                                                </div>
                                            </SimpleBar>

                                            {/* Chimney */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Chimney</h2>
                                                    {renderDetail("Chimney", jobDetails.chimney)}
                                                    {renderDetail("Measurement", jobDetails.chimney_measurement)}
                                                    {renderDetail("Comment", jobDetails.chimney_comment)}
                                                </div>
                                            </SimpleBar>

                                            {/* Roofline */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Roofline</h2>
                                                    {renderDetail("Fascia", jobDetails.fascia)}
                                                    {renderDetail("Fascia Measurement", jobDetails.fascia_measurement)}
                                                    {renderDetail("Soffit", jobDetails.soffit)}
                                                    {renderDetail("Soffit Measurement", jobDetails.soffit_measurement)}
                                                </div>
                                            </SimpleBar>

                                            {/* Rainwater Goods */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Rainwater Goods</h2>
                                                    {renderDetail("Guttering", jobDetails.guttering)}
                                                    {renderDetail("Guttering Replace", jobDetails.guttering_replace)}
                                                    {renderDetail("Guttering Replace Measurement", jobDetails.guttering_replace_measurement)}
                                                    {renderDetail("RWP", jobDetails.rwp)}
                                                    {renderDetail("RWP Replace", jobDetails.rwp_replace)}
                                                    {renderDetail("RWP Replace Measurement", jobDetails.rwp_replace_measurement)}
                                                </div>
                                            </SimpleBar>

                                            {/* Other Works */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Other Works</h2>
                                                    {renderDetail("Completed", jobDetails.other_works_completed)}
                                                    {renderDetail("Needed", jobDetails.other_works_needed)}
                                                </div>
                                            </SimpleBar>

                                            {/* Access */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Access</h2>
                                                    {renderDetail("Access Key", jobDetails.access_key)}
                                                    {renderDetail("Wall Notice", jobDetails.wall_notice)}
                                                </div>
                                            </SimpleBar>

                                            {/* Other Issues */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Other Issues</h2>
                                                    {renderDetail("Issues Present", jobDetails.issues_present)}
                                                    {renderDetail("Comments", jobDetails.issues_comments)}
                                                </div>
                                            </SimpleBar>

                                            {/* Customer Vulnerability */}
                                            <SimpleBar className="mb-4">
                                                <div className="px-4 py-2 rounded-lg shadow transition bg-blue-100 text-blue-800">
                                                    <h2>Customer Vulnerability</h2>
                                                    {renderDetail("Customer Vulnerability", jobDetails.customer_vuln)}
                                                    {renderDetail("Comments", jobDetails.customer_comments)}
                                                </div>
                                            </SimpleBar>
                                        </>
                                    );
                                })()}
                            </div>
                        </div>




                        {/* Image Gallery */}

                        <div className="mt-4 mb-4">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 mb-6 transition-all">
                                {images.map((imgSrc, index) => (
                                    <img
                                        key={index}
                                        src={imgSrc}
                                        alt={`Job photo ${index + 1}`}
                                        className="rounded-lg shadow object-cover w-full h-32 cursor-pointer"
                                        onClick={() => {
                                            setLightboxIndex(index);
                                            setLightboxOpen(true);
                                        }}
                                    />
                                ))}
                            </div>
                        </div>

                        <div>Works Completed:</div>
                        <div className="flex flex-wrap gap-2 mt-6 mb-4">
                            {job.sor_codes
                                ?.filter((code: any) => code.type === "completed")
                                .map((code: any) => (
                                    <button
                                        key={code.sor.id}
                                        onClick={() => handleSelectSor(code.sor.id)}
                                        className={`text-sm font-medium px-4 py-2 rounded-full shadow transition
          ${selectedSorId === code.sor.id
                                                ? "bg-blue-600 text-white"
                                                : "bg-blue-100 text-blue-800 hover:bg-blue-200"
                                            }`}
                                    >
                                        {code.quantity}x {code.sor.id}
                                    </button>
                                ))}


                            <button
                                onClick={() => setOverlayOpen({ open: true, type: "completed" })}
                                className="w-10 h-10 flex items-center justify-center text-xl font-bold text-blue-800 bg-blue-100 hover:bg-blue-200 rounded-full shadow transition"
                                aria-label="Add new SOR"
                            >
                                +
                            </button>

                        </div>

                        <div>Future Works:</div>
                        <div className="flex flex-wrap gap-2 mt-6 mb-4">
                            {job.sor_codes
                                ?.filter((code: any) => code.type === "future")
                                .map((code: any) => (
                                    <button
                                        key={code.sor.id}
                                        onClick={() => handleSelectSor(code.sor.id)}
                                        className={`text-sm font-medium px-4 py-2 rounded-full shadow transition
          ${selectedSorId === code.sor.id
                                                ? "bg-blue-600 text-white"
                                                : "bg-blue-100 text-blue-800 hover:bg-blue-200"
                                            }`}
                                    >
                                        {code.quantity}x {code.sor.id}
                                    </button>
                                ))}


                            <button
                                onClick={() => setOverlayOpen({ open: true, type: "future" })}
                                className="w-10 h-10 flex items-center justify-center text-xl font-bold text-blue-800 bg-blue-100 hover:bg-blue-200 rounded-full shadow transition"
                                aria-label="Add new SOR"
                            >
                                +
                            </button>

                        </div>


                        {isExpanded && selectedSor && (
                            <div className="mt-2 p-4 border rounded-lg bg-gray-50 dark:bg-darkslate text-sm space-y-1">
                                <div><strong>Job Type:</strong> {selectedSor.sor.job_type}</div>
                                <div><strong>Short:</strong> {selectedSor.sor.short}</div>
                                <div><strong>Element:</strong> {selectedSor.sor.element}</div>
                                <div><strong>Work Category:</strong> {selectedSor.sor.work_categories}</div>
                                <div><strong>Work Sub-Category:</strong> {selectedSor.sor.work_sub_categories}</div>
                                <div><strong>Work Sub-Category Attribute:</strong> {selectedSor.sor.work_sub_categories_attributes}</div>
                                <div><strong>Medium:</strong> {selectedSor.sor.medium}</div>
                                <div className="pt-4 flex justify-center">
                                    <button
                                        onClick={() => setConfirmDeleteOpen(true)}
                                        className="text-sm bg-red-600 text-white px-4 mr-2 py-2 rounded shadow hover:bg-red-700 transition"
                                    >
                                        Delete Code
                                    </button>
                                    <button
                                        onClick={() => changeState(true, selectedSor)}
                                        className="text-sm bg-blue-600 text-white px-4 ml-2 py-2 rounded shadow hover:bg-blue-700 transition"
                                    >
                                        Change Quantity
                                    </button>
                                </div>
                            </div>
                        )}
                        <div className="pt-4 pb-4 flex justify-center">
                            <button
                                onClick={() => setConfirmSubmitOpen(true)}
                                className="text-sm bg-red-600 text-white px-10 py-2 rounded shadow hover:bg-red-700 transition"
                            >
                                Submit to Propeller
                            </button>
                        </div>
                        <div>System ID: {jobId}</div>
                    </div>
                </SimpleBar>
            </div>
            <SOROverlay
                open={overlayOpen.open}
                onClose={() => setOverlayOpen({ open: false, type: null })}
                jobId={jobId}
                sorType={overlayOpen.type}
                existingSorCodes={job?.sor_codes.map((code: any) => code.sor.id.toString()) || []}
                onUpdated={() => {
                    refreshPage();
                }}
            />
            {confirmDeleteOpen && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white dark:bg-darkslate p-6 rounded-lg shadow-lg max-w-sm text-center space-y-4">
                        <p>Are you sure you want to delete this SOR code?</p>
                        <div className="flex justify-center gap-4">
                            <button
                                onClick={handleDeleteSor}
                                className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
                            >
                                Yes, Delete
                            </button>
                            <button
                                onClick={() => setConfirmDeleteOpen(false)}
                                className="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {confirmSubmitOpen && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white dark:bg-darkslate p-6 rounded-lg shadow-lg max-w-sm text-center space-y-4">
                        <p>Are you sure you want to submit those codes to Propeller?</p>
                        {/* Dropdown Menu */}
                        <div>
                            <label htmlFor="clientSelect" className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                                Choose a client:
                            </label>
                            <select
                                id="clientSelect"
                                value={selectedClient}
                                onChange={(e) => setSelectedClient(e.target.value)}
                                className="w-full border border-gray-300 dark:border-gray-600 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-darkslate dark:text-white"
                            >
                                <option value="">-- Select Client --</option>
                                <option value="UPS">UPS</option>
                                <option value="Livv Housing Group">Livv Housing Group</option>
                                <option value="Evolve">Evolve</option>
                                <option value="Evolve (D&M)">Evolve (D&M)</option>
                                <option value="HMS">HMS</option>
                                <option value="Sovini">Sovini</option>
                            </select>
                        </div>

                        <div className="flex justify-center gap-4">
                            <button
                                onClick={handleSubmitSor}
                                disabled={!selectedClient}
                                className={`px-4 py-2 rounded text-white ${selectedClient
                                    ? 'bg-red-600 hover:bg-red-700'
                                    : 'bg-gray-400 cursor-not-allowed'
                                    }`}
                            >
                                Yes, Submit
                            </button>

                            <button
                                onClick={() => setConfirmSubmitOpen(false)}
                                className="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {setQuantity.state && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white dark:bg-darkslate p-6 rounded-lg shadow-lg max-w-sm text-center space-y-4">
                        <p>Please set a new quantity.</p>
                        <input
                            id="quantity"
                            type="number"
                            required
                            className="bg-white text-black form-control form-rounded-xl"
                            onChange={(e) => setQuant(e.target.valueAsNumber)}
                        />
                        <div className="flex justify-center gap-4">
                            <button
                                onClick={() => changeQuantity(quant)}
                                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                            >
                                Confirm
                            </button>
                            <button
                                onClick={() => changeState(false, "")}
                                className="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {lightboxOpen && (
                <Lightbox
                    open={lightboxOpen}
                    close={() => setLightboxOpen(false)}
                    slides={images.map((src) => ({ src }))}
                    index={lightboxIndex}
                    plugins={[Zoom]}
                />
            )}
        </div>
    );
};

export default JobView;
