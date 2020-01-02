$(document).ready(function() {
             $('#getparams').click(function() {
                 var data = '';
                 $.ajax({
                     type: 'POST',
                     url: '/image_upload/',
                     data: data,
                     processData: false,
                     contentType: false,
                     success: function(data) {
                        $.each(data.answers, function(i, val) {
                          console.log(i)
                          $("#" + i).html(val[0] + " - " + val[1]);
                        });
                        console.log(data.answers)
                    }
                 })
             });

             $('#merge').click(function() {
                 var data = '';
                 $.ajax({
                     type: 'POST',
                     url: '/merge/',
                     data: data,
                     processData: false,
                     contentType: false,
                     success: function(data) {
                        $.each(data.merge, function(n, line) {
                          $("#merge_text").append("<p>"+line+"</p>")
                        })
                     }
                 })
             });
         });