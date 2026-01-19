import { useState } from "react";

interface SorOverlayProps {
    open: boolean;
    onClose: () => void;
    jobId: string | null;
    sorType: "completed" | "future" | null;
    existingSorCodes: string[];
    onUpdated?: () => void;
}

interface SorRow {
    id: number;
    job_type: string;
    short: string;
    medium: string;
    element: string;
    work_categories: string;
    work_sub_categories: string;
    work_sub_categories_attributes: string;
}

// Value options for dropdown columns
const JOB_TYPE_OPTIONS = [
    "AIRBRICK",
    "ASPHALT",
    "BOXED END",
    "BOW/BAY",
    "CHIMNEY",
    "CHIMNEY BREAST",
    "CLADDING",
    "CLIENT INSPECTION",
    "COLLAR/STRUT",
    "DOWNPIPE",
    "DORMERS",
    "FASCIA",
    "FASCIA/BARGE",
    "FASCIA/SOFFIT/BARGE",
    "FELT",
    "FILLET",
    "FIREWALL",
    "FLASHING",
    "FLAT ROOF OUTLET",
    "FLOOR",
    "GREEN ROOF",
    "GUTTER",
    "HANGER",
    "HIP",
    "HOPPER",
    "INSULATION",
    "JOIST",
    "PARTITION",
    "PLATE",
    "PORCH",
    "RAFTER",
    "RIDGE",
    "ROOF",
    "ROOF BOARDING",
    "ROOF COVERING",
    "ROOF TILE",
    "ROOFING",
    "SHEETING",
    "SLATE",
    "SOAKER",
    "SOFFIT",
    "SPECIAL",
    "TILE",
    "VALLEY",
    "VENT",
    "VENTILATOR",
    "VERGE",
    "VERTICAL COVERING",
    "renew through valley tile"
];

const ELEMENT_OPTIONS = [
    "BRICKWORK",
    "CARPENTRY AND JOINERY",
    "DRAINAGE.1",
    "PAINTING AND DECORATING",
    "PLUMBING",
    "ROOFING",
    "SPECIALIST TREATMENTS"
];

const WORK_CATEGORY_OPTIONS = [
    "Airbricks and Vents",
    "Asphalt Roofing",
    "Brick/Block Walling",
    "Chimneys",
    "Clay/Concrete Roof Tiling",
    "External Cladding",
    "Fascia Soffit and Bargeboards",
    "Felt Roofing",
    "Floors Roofs and Partitions",
    "Green Rooing",
    "Leadwork",
    "Miscellaenous Works",
    "Miscellaneous Works",
    "Preparation - External and Internal",
    "Rainwater Gutters",
    "Rainwater Pipework",
    "Remidial Works",
    "Sheet Roofing",
    "Slate Roofing",
    "Sundry Works",
    "Timber Treatment",
    "Vertical Coverings"
];

export default function SorOverlay({ open, onClose, jobId, sorType, existingSorCodes, onUpdated }: SorOverlayProps) {
    const [rows, setRows] = useState<SorRow[]>([]);
    const [limit, setLimit] = useState(10);
    const [sortBy] = useState("id");
    const [sortOrd, setSortOrd] = useState<"asc" | "desc">("asc");
    const [searchWord, setSearchWord] = useState("");
    const [searchColumn, setSearchColumn] = useState("id");
    const [quant, setQuant] = useState<number | null>(null);
    const [setQuantity, setConfirmQuantityOpen] = useState({ state: false, row: "" });

    const changeState = (newState: boolean) => {
        setConfirmQuantityOpen(prev => ({ ...prev, state: newState }));
    };

    const changeRow = (newRow: string) => {
        changeState(true);
        setConfirmQuantityOpen(prev => ({ ...prev, row: newRow }));
    };

    const handleAddSorCode = (sorId: string, quantity: number | null) => {
        if (!jobId) return;
        if (quant == null || isNaN(quant)) {
            alert("Please enter a valid quantity.");
            return;
        }

        const updatedCodes = [...new Set([sorId])];
        fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/jobs/change_sor?job_id=${jobId}`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${localStorage.getItem("token") || ""}`,
            },
            body: JSON.stringify({ new_sor_codes: updatedCodes, quantity, type: sorType }),
        })
            .then(res => {
                if (!res.ok) throw new Error("Failed to update SOR codes");
                return res.json();
            })
            .then(() => {
                onClose();
                onUpdated?.();
            })
            .catch(err => {
                console.error("SOR update error:", err);
                alert("Failed to update SOR codes");
            });

        changeState(false);
    };

    const fetchRows = () => {
        const params = new URLSearchParams({
            limit: limit.toString(),
            sort_by: sortBy,
            sort_ord: sortOrd,
            column: searchColumn,
        });
        if (searchWord.trim()) params.append("search", searchWord.trim());

        fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/sor/rows?${params.toString()}`, {
            headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` },
        })
            .then(res => {
                if (!res.ok) throw new Error("Network response was not ok");
                return res.json();
            })
            .then(data => setRows(data))
            .catch(err => console.error("Fetch error:", err));
    };

    if (!open) return null;

    // Determine dropdown options if applicable
    const getColumnOptions = () => {
        if (searchColumn === "job_type") return JOB_TYPE_OPTIONS;
        if (searchColumn === "element") return ELEMENT_OPTIONS;
        if (searchColumn === "work_categories") return WORK_CATEGORY_OPTIONS;
        return null;
    };

    const columnOptions = getColumnOptions();

    return (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center">
            <div className="bg-white rounded-xl shadow-xl w-full max-w-3xl max-h-[90vh] p-6 overflow-y-auto relative">
                {/* Close button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-gray-400 hover:text-gray-700 text-xl font-bold"
                >
                    &times;
                </button>

                <h2 className="text-xl font-semibold mb-4">Browse SOR Database</h2>

                {/* Controls */}
                <div
                    className="flex flex-wrap gap-2 mb-4"
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            e.preventDefault();
                            fetchRows();
                        }
                    }}
                >
                    {/* Column selector */}
                    <div className="flex flex-col">
                        <label className="text-sm font-medium mb-1">Category</label>
                        <select
                            value={searchColumn}
                            onChange={(e) => {
                                setSearchColumn(e.target.value);
                                setSearchWord(""); // Reset search when changing column
                            }}
                            className="border px-3 py-2 rounded-md"
                        >
                            <option value="id">ID</option>
                            <option value="job_type">Job Type</option>
                            <option value="element">Element</option>
                            <option value="work_categories">Work Category</option>
                        </select>
                    </div>

                    {/* Search input or dropdown */}
                    <div className="flex flex-col">
                        <label className="text-sm font-medium mb-1">Query</label>
                        {columnOptions ? (
                            <select
                                value={searchWord}
                                onChange={(e) => setSearchWord(e.target.value)}
                                className="border px-3 py-2 rounded-md w-48"
                            >
                                <option value="">Select...</option>
                                {columnOptions.map((opt) => (
                                    <option key={opt} value={opt}>
                                        {opt}
                                    </option>
                                ))}
                            </select>
                        ) : (
                            <input
                                type="text"
                                placeholder="Search..."
                                value={searchWord}
                                onChange={(e) => setSearchWord(e.target.value)}
                                className="border px-3 py-2 rounded-md w-48"
                            />
                        )}
                    </div>

                    <div className="flex flex-col">
                        <label className="text-sm font-medium mb-1">Limit</label>
                        <input
                            type="number"
                            min="1"
                            value={limit}
                            onChange={(e) => setLimit(Number(e.target.value))}
                            className="border px-3 py-2 rounded-md w-20"
                        />
                    </div>

                    <div className="flex flex-col">
                        <label className="text-sm font-medium mb-1">Order</label>
                        <select
                            value={sortOrd}
                            onChange={(e) => setSortOrd(e.target.value as "asc" | "desc")}
                            className="border px-3 py-2 rounded-md"
                        >
                            <option value="asc">Asc</option>
                            <option value="desc">Desc</option>
                        </select>
                    </div>

                    <div className="flex flex-col">
                        <label className="text-sm font-medium mb-1">&nbsp;</label>
                        <button
                            onClick={fetchRows}
                            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition h-[42px]"
                        >
                            Search
                        </button>
                    </div>
                </div>

                {/* Results */}
                <div className="space-y-2">
                    {rows.length > 0 ? (
                        rows.map((row) => {
                            const alreadyAdded = existingSorCodes.includes(row.id.toString());
                            return (
                                <div
                                    key={row.id}
                                    className={`p-3 rounded-lg border shadow-sm transition
                                        ${alreadyAdded ? "bg-gray-200 text-gray-500 cursor-not-allowed" : "bg-gray-50 hover:bg-blue-50 cursor-pointer"}`}
                                    onClick={() => !alreadyAdded && changeRow(row.id.toString())}
                                >
                                    <p className="font-medium">{row.id}</p>
                                    <p className="font-medium">{row.job_type}</p>
                                    <p className="font-medium">{row.short}</p>
                                </div>
                            );
                        })
                    ) : (
                        <p className="text-gray-500 italic">No results found.</p>
                    )}
                </div>
            </div>

            {/* Quantity dialog */}
            {
                setQuantity.state && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                        <div className="bg-white p-6 rounded-lg shadow-lg max-w-sm text-center space-y-4">
                            <p>Please set the quantity for this code.</p>
                            <input
                                id="quantity"
                                type="number"
                                step="0.01"
                                min="0"
                                required
                                className="border px-3 py-2 rounded-md w-32"
                                onChange={(e) => setQuant(parseFloat(e.target.value))}
                                value={quant || ""}
                            />
                            <div className="flex justify-center gap-4">
                                <button
                                    onClick={() => handleAddSorCode(setQuantity.row, quant)}
                                    className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
                                >
                                    Confirm
                                </button>
                                <button
                                    onClick={() => changeState(false)}
                                    className="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                )
            }
        </div >
    );
}
