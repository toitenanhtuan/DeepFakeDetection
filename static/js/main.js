document.addEventListener('DOMContentLoaded', function() {
    // File upload validation
    const videoInput = document.getElementById('video');
    if (videoInput) {
        videoInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            const fileType = file.type;
            const validTypes = ['video/mp4', 'video/avi', 'video/quicktime'];
            
            if (!validTypes.includes(fileType)) {
                alert('Please upload a valid video file (MP4, AVI, or MOV)');
                e.target.value = '';
            }
        });
    }

    // Add loading indicator during upload
    const uploadForm = document.querySelector('form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Analyzing...';
            submitButton.disabled = true;
        });
    }
});
