$(document).ready(function() {
             $('#getparams').click(function() {
                 var data = '';
                 $.ajax({
                     type: 'POST',
                     url: '/image_upload/',
                     data: data,
                     processData: false,
                     contentType: false,
                     success: console.log("success")
                 })
             });
         });