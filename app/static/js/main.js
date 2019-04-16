var i = 0;
files = null;
var url = null;

function change(file){
files = file.files;
let biData = [];
biData.push(files[i]);
url = window.URL.createObjectURL(new Blob(biData,{type:"application/zip"}));
$('#originImage').attr('src',url);
}

function nextImage(){
i += 1;
if (i<files.length){
let biData = [];
biData.push(files[i]);
url = window.URL.createObjectURL(new Blob(biData,{type:"application/zip"}));
alert(url);
$('#originImage').attr('src',url);
}
}

function upload(){
var formdata = new FormData();
formdata.append(files[i].name,files[i]);
formdata.append('name',files[i].name);
alert(i,files[i].name)
alert(formdata['name'])
$.ajax({
    url:"http://127.0.0.1:5000/upload_image",
    type:'POST',
    cache:false,
    data:formdata,
    processData:false,
    contentType:false,
    success: function(datas){
    $('#outputImage').attr('src','static/output/'+files[i].name+'?Math.round(Math.random()*1000)');
    alert(datas['data']['box_num']);
    var xx = datas['data']['annotations'];
    alert('bb'+xx[1]['text']);

    box_num = parseInt(datas['data']['box_num']);
    lines = box_num//2;
    innerhtml = '';
   for(var i = 0;i<lines;i++){
   innerhtml += '<tr>'+
                '<td>'+xx[i*2]['text']+'</td>'+
                '<td>'+xx[1]['bbox']+'</td>'+
                '<td>'+xx[i*2]['text']+'</td>'+
                '<td>'+xx[1]['bbox']+'</td>'+
            '</tr>';
   }
   $('#dataTbody').text(innerhtml);
    },
    error:function(){
    alert('error');
    }
    });
}