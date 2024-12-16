document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.querySelector('input[type="file"]');
    fileInput.addEventListener('change', function(e) {
        const fileName = e.target.files[0].name;
        console.log('Selected file:', fileName);
    });
});