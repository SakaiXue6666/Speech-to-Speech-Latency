fetch('/openapi.json').then(r=>r.json()).then(function(){
  setTimeout(function(){
    var results = [];
    document.querySelectorAll('.responses-wrapper').forEach(function(w, i){
      var info = {wrapper: i, children: []};
      for(var c = 0; c < w.children.length; c++){
        var ch = w.children[c];
        info.children.push({tag: ch.tagName, class: ch.className, text: ch.textContent.substring(0,80)});
      }
      results.push(info);
    });
    document.title = JSON.stringify(results);
  }, 2000);
});
