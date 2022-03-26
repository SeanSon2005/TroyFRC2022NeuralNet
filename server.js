var fs = require('fs');
var http = require('http')

var reqListener = function(req, res) {
    if(req.method == "POST") {
        console.log("POST REQ")
        var body = "";
        req.on("data", function(data) {
            body += data;
        });
        req.on("end", function() {
            console.log("data received");
            fs.writeFileSync("info.txt", body);
        });
    }
}

http.createServer(reqListener).listen(6969);