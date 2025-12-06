// Function to handle PDF export
document.addEventListener('DOMContentLoaded', function() {
    const exportPdfBtn = document.getElementById('exportPdf');
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener('click', function(e) {
            e.preventDefault();
            // Using html2pdf library for PDF generation
            const element = document.body;
            const opt = {
                margin: 10,
                filename: 'portfolio-optimization-results.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2 },
                jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
            };

            // Check if html2pdf is available
            if (typeof html2pdf !== 'undefined') {
                html2pdf().from(element).set(opt).save();
            } else {
                alert('PDF generation library not loaded. Please try again.');
                console.error('html2pdf not found');
            }
        });
    }
});
