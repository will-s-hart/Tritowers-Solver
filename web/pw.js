// Serial Python worker. Fonts are public; photos and game state stay in this browser.
import { loadPyodide } from '/pyodide/pyodide.mjs';
let py,vision;
function loadVision(){
  if(!vision){
    const attempt=py.loadPackage(['numpy','pillow','opencv-python']);vision=attempt;
    attempt.catch(()=>{if(vision===attempt)vision=null});
  }
  return vision;
}
const ready=(async()=>{
  py=await loadPyodide({indexURL:'/pyodide/'});
  const response=await fetch('/app.zip');if(!response.ok)throw new Error('Could not load the app');
  py.unpackArchive(await response.arrayBuffer(),'zip',{extractDir:'/app'});
  py.runPython("import sys, json\nsys.path.insert(0, '/app')\nimport web_core as w");
  // Start fetching vision immediately, before the user picks a photograph.
  loadVision();
})();
let queue=Promise.resolve(),latestPhoto=null;
// Once idle, build the font caches the first photo read would otherwise pay for.
queue=queue.then(async()=>{try{await ready;await loadVision();py.runPython('w.warm_vision()')}catch(e){}});
const waiting=new Set(),cancelled=new Set();
self.onmessage=({data})=>{
  if(data.cancel!==undefined){if(waiting.has(data.cancel))cancelled.add(data.cancel);return}
  waiting.add(data.id);
  if(data.route==='/api/photo')latestPhoto=data.id;
  queue=queue.then(async()=>{
    const {id,route,body,bytes,corners}=data;
    const obsolete=()=>{
      if(!cancelled.has(id)&&!(route==='/api/photo'&&id!==latestPhoto))return false;
      self.postMessage({id,ok:false,error:'This request was cancelled or replaced by a newer photo.'});return true;
    };
    try{
      if(obsolete())return;
      await ready;if(obsolete())return;let out;
      if(route==='/api/photo'){
        await loadVision();if(obsolete())return;
        py.FS.writeFile('/tmp/in.bin',new Uint8Array(bytes));
        py.globals.set('tt_corners',corners||'');
        try{out=py.runPython("json.dumps(w.api_photo(open('/tmp/in.bin','rb').read(), tt_corners))")}
        finally{py.FS.unlink('/tmp/in.bin')}
      }else{
        py.globals.set('tt_body',JSON.stringify(body||{}));
        const fn={'/api/new':'api_new','/api/act':'api_act','/api/solve':'api_solve','/api/geo':'api_geo','/health':'health'}[route];
        if(!fn)throw new Error('Unknown route');
        out=py.runPython(`json.dumps(w.${fn}(${['api_geo','health'].includes(fn)?'':'json.loads(tt_body)'}))`);
      }
      self.postMessage({id,ok:true,json:out});
    }catch(e){self.postMessage({id,ok:false,error:'The local engine could not finish. Reload and try again.'})}
    finally{waiting.delete(id);cancelled.delete(id)}
  });
};
