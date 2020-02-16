var classifyUrl = '/api/v1/classify';

function resetFileProgress(){
  $('#progress .progress-bar').css('width', '0%');
  $('#progress').hide();
}

function resetFiles(){
  $('#files').empty();
}

function startFileUpload(){
  $('#progress').show();
}

function resetClassification(){
  $('#result').empty();
}

function resetClassificationBar(){
  $('#confidence').empty();
}
function resetPredictionImage(){
  $('#painting-box').empty();
}

function animateConfidence(data){
  $('#confidence').each(function(){
    var progress = $(this);
    var percent = progress.data('percent');
    for (var x = 0; x < data.length; x++) {
      console.log(data[x].percent);
      percent = data[x].percent;
    }

    var bar = new ProgressBar.Line(this, {
      color: '#6fda00',
      strokeWidth: 4,
      easing: 'easeInOut',
      duration: 1400,
      trailColor: '#eee',
      trailWidth: 1,
      svgStyle: {width: '100%', height: '100%'},
      text: {
        style: null,
        autoStyleContainer: false
      },
      step: (state, bar) => {
        bar.setText(Math.round(bar.value() * 100) + ' % PPC');
      }
    });

    bar.animate(percent);
  });
}

function printClassificationResult(data){
  for (var x = 0; x < data.length; x++) {
    console.log(data);
    console.log(data[x].caption);
    console.log(data[x].image_path);
    $('<p/>').text(data[x].caption).appendTo('#result');
    var img = new Image();
    img.src = data[x].image_path;
    img.setAttribute("class", "img-painting");
    document.getElementById("painting-box").appendChild(img);
  }

  animateConfidence(data)
}

$(document).ready(function(){
  window.sr = ScrollReveal();
  sr.reveal('.reveal');
});

$(document).ready(function(){
  $('#fileupload').fileupload({
    url: classifyUrl,
    cache: false,
    dataType: 'json',
    send: function (e, data) {
      resetFiles();
      resetClassification();
      resetClassificationBar();
      resetPredictionImage();
      startFileUpload();

      var f = data.files[0];
      var reader = new FileReader();

      reader.onload = (function(file) {
        return function(e) {
          var img = $('<img>');
          img.attr('src', e.target.result);
          img.attr('title', file.name);
          img.addClass('img-painting');
          img.addClass('center-block');
          img.appendTo('#files');
        };
      })(f);

      reader.readAsDataURL(f);
    },
    progressall: function(e, data) {
      var progress = parseInt(data.loaded / data.total * 100, 10);
      $('#progress .progress-bar').css('width', progress + '%');
    },
    done: function (e, data) {
      resetFileProgress();
      printClassificationResult(data.result);
    }
  })
  .prop('disabled', !$.support.fileInput).parent().addClass($.support.fileInput ? undefined : 'disabled');
});