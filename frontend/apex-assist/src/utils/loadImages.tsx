export const loadImagesByJobId = async (jobId: string): Promise<string[]> => {
    try {
        // Fetch the list of image filenames from the backend
        const response = await fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/jobs/images/${jobId}`);
        if (!response.ok) throw new Error("Failed to fetch image list");

        const filenames: string[] = await response.json();

        return filenames;
    } catch (err) {
        console.error("Error loading images for job:", jobId, err);
        return [];
    }
};