const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

const page=fs.readFileSync(path.join(__dirname,'../../apps/dashboard_static/training_worklog.html'),'utf8');
const context=vm.createContext({});
for(const id of ['markdown-it-vendor','worklog-renderer']){
  const source=page.match(new RegExp('<script id="'+id+'">([\\s\\S]*?)</script>'));
  assert.ok(source,`Missing ${id}`);
  vm.runInContext(source[1],context,{timeout:2000});
}
const {markdown,worklogSections,sectionMatches}=vm.runInContext('({markdown,worklogSections,sectionMatches})',context);
const html=source=>markdown.renderer.render(worklogSections(source)[0],markdown.options,{});

test('real Markdown tables retain alignment, code, and escaped pipes',()=>{
  const result=html('### 测速\n\n| 方法 | 秒 |\n| --- | ---: |\n| **原版** | 708.41 |\n| `a\\|b` | 598.20 |');
  assert.match(result,/<table>/);
  assert.match(result,/<thead>/);
  assert.equal((result.match(/<tr>/g)||[]).length,3);
  assert.match(result,/<td style="text-align:right">598\.20<\/td>/);
  assert.match(result,/<strong>原版<\/strong>/);
  assert.match(result,/<code>a\|b<\/code>/);
});

test('fenced headings and quoted headings stay inside their log section',()=>{
  const source='### 较早\n\n```python\n## code heading\nprint(1)\n```\n\n> ## quoted heading\n\n### 最新\n\n新记录';
  const groups=worklogSections(source);
  assert.equal(groups.length,2);
  assert.equal(groups[0][1].content,'最新');
  const earlier=markdown.renderer.render(groups[1],markdown.options,{});
  assert.match(earlier,/<pre><code class="language-python">## code heading/);
  assert.match(earlier,/<blockquote>/);
  assert.ok(sectionMatches(groups[1],'code heading'));
  assert.ok(!sectionMatches(groups[0],'code heading'));
});

test('lists, links, quotes, and inline code render without active raw HTML',()=>{
  const result=html('### 标记\n\n- 项目 `one`\n- 项目 **two**\n\n> 引用\n\n[文档](https://example.com/)\n\n<img src=x onerror="globalThis.attacked=true">\n\n<script>globalThis.attacked=true</script>\n\n[坏链接](javascript:alert(1))');
  assert.match(result,/<ul>/);
  assert.match(result,/<code>one<\/code>/);
  assert.match(result,/<blockquote>/);
  assert.match(result,/href="https:\/\/example.com\/"/);
  assert.doesNotMatch(result,/<script|<img|href="javascript:/);
  assert.match(result,/&lt;script&gt;/);
});

test('a LAN draft keeps its message ID across a failed send and reload',async()=>{
  const controls=page.match(/<script id="worklog-controls">([\s\S]*?)<\/script>/)[1];
  const {webcrypto}=require('node:crypto');
  const storage=new Map(),requests=[];
  let snapshot={enabled:true,can_send:true,entries:[]};
  function openPage(){
    const fields=new Map();
    function getElementById(id){
      if(!fields.has(id))fields.set(id,{value:'',replaceChildren(){}});
      return fields.get(id);
    }
    const browser=vm.createContext({
      document:{getElementById},setInterval(){},
      crypto:{getRandomValues:bytes=>webcrypto.getRandomValues(bytes)},
      sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
      async fetch(url,options){
        if(url==='/api/worklog')return {ok:true,json:async()=>snapshot};
        assert.equal(url,'/api/messages');
        const payload=JSON.parse(options.body);
        requests.push(payload);
        if(requests.length===1)throw Error('lost response');
        return {ok:true,json:async()=>({id:'USER-'+payload.id,path:'worklog/2026-11-18.md'})};
      }
    });
    vm.runInContext(controls,browser);
    return {field:getElementById,refresh:()=>vm.runInContext('refresh()',browser),
      submit:()=>getElementById('message-form').onsubmit({preventDefault(){}})};
  }
  let ui=openPage();
  await ui.refresh();
  assert.equal(ui.field('send').disabled,true);
  snapshot={...snapshot,branch:'main',commit:'abc',send_token:'test',updated_at:Date.now()/1000};
  await ui.refresh();
  assert.equal(ui.field('send').textContent,'发送到 main');
  ui.field('message').value='同步后的留言';
  ui.field('message').oninput();
  await ui.submit();
  assert.equal(ui.field('message').value,'同步后的留言');
  assert.match(requests[0].id,/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  ui=openPage();
  await ui.refresh();
  await ui.submit();
  assert.deepEqual(requests[1],requests[0]);
  assert.equal(ui.field('message').value,'');
  assert.ok(ui.field('send-result').textContent.includes('USER-'+requests[0].id));
});
