const dropzoneBox = document.getElementsByClassName("dropzone-box")[0];

const inputFiles = document.querySelectorAll(
  ".dropzone-area input[type='file']"
);

const inputElement = inputFiles[0];

const dropZoneElement = inputElement.closest(".dropzone-area");

inputElement.addEventListener("change", (e) => {
  if (inputElement.files.length) {
    updateDropzoneFileList(dropZoneElement, inputElement.files[0]);
  }
});

dropZoneElement.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZoneElement.classList.add("dropzone--over");
});

["dragleave", "dragend"].forEach((type) => {
  dropZoneElement.addEventListener(type, (e) => {
    dropZoneElement.classList.remove("dropzone--over");
  });
});

dropZoneElement.addEventListener("drop", (e) => {
  e.preventDefault();

  if (e.dataTransfer.files.length) {
    inputElement.files = e.dataTransfer.files;

    updateDropzoneFileList(dropZoneElement, e.dataTransfer.files[0]);
  }

  dropZoneElement.classList.remove("dropzone--over");
});

const updateDropzoneFileList = (dropzoneElement, file) => {
  let dropzoneFileMessage = dropzoneElement.querySelector(".file-info");

  dropzoneFileMessage.innerHTML = `
        ${file.name}, ${file.size} bytes
    `;
};

dropzoneBox.addEventListener("reset", (e) => {
  let dropzoneFileMessage = dropZoneElement.querySelector(".file-info");

  dropzoneFileMessage.innerHTML = `No Files Selected`;
});

dropzoneBox.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("upload-file").files[0];
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('https://the-analyser.onrender.com/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    console.log(data);
  } catch (error) {
    console.error('Error:', error);
  }
});