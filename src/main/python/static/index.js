function previewFile1(){
  var preview = document.querySelector('#img1'); //selects the query named img
  var file    = document.querySelector('input[id=file-picker1]').files[0]; //sames as here
  var reader  = new FileReader();

  reader.onloadend = function () {
    preview.src = reader.result;
  }

  if (file) {
    reader.readAsDataURL(file); //reads the data as a URL
  } else {
    preview.src = "";
  }

  //reset fields
  document.getElementById("encoding1_txt").innerHTML = "";
  document.getElementById("encoding1").value = "";
  document.getElementById("result_placeholder").innerHTML = "";
}

function previewFile2() {
  var preview = document.querySelector('#img2'); //selects the query named img
  var file    = document.querySelector('input[id=file-picker2]').files[0]; //sames as here
  var reader  = new FileReader();

  reader.onloadend = function () {
    preview.src = reader.result;
  }

  if (file) {
    reader.readAsDataURL(file); //reads the data as a URL
  } else {
    preview.src = "";
  }

  //reset fields
  document.getElementById("encoding2_txt").innerHTML = "";
  document.getElementById("encoding2").value = "";
  document.getElementById("result_placeholder").innerHTML = "";
}