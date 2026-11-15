from .run_device import artifact_directory
from pathlib import Path
import ctypes,hashlib,json
from .graph_runtime import torch,ptr,lib,check
P=artifact_directory()

def build_deck_artifacts(inputs,*,snapshot,canonical_header,nvrtc,nvcc,directory=None,kinds=('observer','rule','builder'),architecture='sm_120'):
 """Original NVRTC observer and nvcc builder, with explicit consumer inputs.

 The simulator stays in the qualified external snapshot. No generated game
 source, checkpoint or binary is copied into the repository.
 """
 import subprocess,time
 from .deck_inputs import portable_compact_header
 output=Path(directory or P/'deck');output.mkdir(parents=True,exist_ok=True)
 source=Path(__file__).resolve().parent;snapshot=Path(snapshot)
 original=snapshot/'devicecompact141';header=snapshot/'deviceflex141/device_std.h'
 digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert digest(header)=='b4295d78a2df81e696177d5f97f2d5a1b07de6aa09c5d0570831a6335b727e51'
 assert (original/'generated/include/combat/BattleContext.h').is_file()
 (output/'deck_schema.h').write_text(inputs.header(),newline='\n')
 (output/'compact-observation-device.h').write_text(portable_compact_header(canonical_header),newline='\n')
 compiled={}
 for kind,filename in [('observer','observer_entry.cu'),('rule','rules_entry.cu')]:
  if kind not in kinds:continue
  generated=snapshot/('deviceflex141/generated' if kind=='rule' else 'devicecompact141/generated')
  if kind=='rule':
   manifest=json.loads((snapshot/'deviceflex141/outlined-source-manifest.json').read_text())
   assert len(manifest['selected_functions'])==185 and manifest['factory_count']==99
   for name,row in manifest['overrides'].items():assert digest(generated/name)==row['sha256'],('original outlined method differs',name)
   # Retain the qualified outlined simulator. Add only the native queue's
   # data-only opening continuation and its unchanged Actions consumers.
   simulator=Path(canonical_header).resolve().parents[1]
   continuation=output/'continuation-simulator'
   continuation_sources={}
   for relative,marker,anchor in [
    ('include/combat/Actions.h','PORTABLE OPENING ACTION','    class CardQueueItem;'),
    ('include/combat/ActionQueue.h','PENDING ACTION ACCESS','        [[nodiscard]] int getCapacity() const;'),
    ('include/combat/BattleContext.h','HEAVENLY PUNISHMENT','        std::bitset<32> miscBits;'),
    ('src/combat/Actions.cpp','PORTABLE OPENING ACTION','using namespace sts;')]:
    native=simulator/relative;text=native.read_text();begin='// BEGIN '+marker;end='// END '+marker
    section=text[text.index(begin):text.index(end)+len(end)]
    section=section.replace('throw std::logic_error("unknown opening continuation");',
                            'gpu_fail(__LINE__); return {};')
    section=section.replace('throw std::logic_error("pending Stasis return lost its captured card");',
                            'gpu_fail(__LINE__); return;')
    original_text=(generated/relative).read_text();assert original_text.count(anchor)==1,(relative,anchor)
    target=continuation/relative;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(original_text.replace(anchor,anchor+'\n'+section))
    continuation_sources[relative]=dict(native_sha256=digest(native),generated_sha256=digest(target))
   relative='src/combat/BattleContext.cpp'
   original_text=(generated/relative).read_text()
   native=(simulator/relative).read_text()
   old='        ++player.cardsPlayedThisTurn;'
   new=old+'\n        if (player.cardsPlayedThisTurn > 100 && applyHeavenlyPunishment()) return;'
   assert original_text.count(old)==1 and new in native
   original_text=original_text.replace(old,new)
   start=original_text.index('void BattleContext::afterMonsterTurns()')
   head,tail=original_text[:start],original_text[start:]
   new='    if (turn >= 20 && applyHeavenlyPunishment()) return;'
   assert new in native
   original_text=head+tail.replace('    ++turn;', '    ++turn;\n'+new, 1)
   old='addToBot(Actions::OnAfterCardUsed());'
   replacement='addToBot(OpeningAction::create(OpeningAction::AfterCardUsed));'
   assert original_text.count(old)==1 and replacement in (simulator/relative).read_text()
   original_text=original_text.replace(old,replacement)
   old='case CardId::BURNING_PACT:\n            addToBot( Actions::ChooseExhaustOne() );\n            addToBot( Actions::DrawCards(up ? 3 : 2) );'
   replacement=old.replace('Actions::DrawCards(up ? 3 : 2)', 'OpeningAction::create(OpeningAction::Draw, up ? 3 : 2)')
   assert original_text.count(old)==1 and replacement in (simulator/relative).read_text()
   target=continuation/relative;target.parent.mkdir(parents=True,exist_ok=True)
   original_text=original_text.replace(old,replacement)
   old='Actions::MakeTempCardInDrawPile(CardInstance(CardId::DAZED), 1, true)'
   replacement='OpeningAction::create(OpeningAction::TempDraw, int(CardId::DAZED), 1, true)'
   assert original_text.count(old)==3 and (simulator/relative).read_text().count(replacement)==3
   original_text=original_text.replace(old,replacement)
   begin=original_text.index('void BattleContext::callEndOfTurnActions()')
   end=original_text.index('void BattleContext::onTurnEnding()',begin)
   section=original_text[begin:end]
   section=section.replace('Actions::GainBlock(', 'OpeningAction::create(OpeningAction::Block, ')
   section=section.replace('Actions::DamageAllEnemy(52)', 'OpeningAction::create(OpeningAction::DamageAll, 52)')
   section=section.replace('Actions::TriggerEndOfTurnOrbsAction()', 'OpeningAction::create(OpeningAction::EndTurnOrbs)')
   target.write_text(original_text[:begin]+section+original_text[end:])
   continuation_sources[relative]=dict(native_sha256=digest(simulator/relative),generated_sha256=digest(target))
   relative='src/combat/Monster.cpp'
   original_text=(generated/relative).read_text()
   assert original_text.count('Actions::MonsterGainBlock(')==3
   assert original_text.count('Actions::ReactiveRollMove(idx)')==1
   original_text=original_text.replace('Actions::MonsterGainBlock(', 'OpeningAction::create(OpeningAction::MonsterBlock, ')
   original_text=original_text.replace('Actions::ReactiveRollMove(idx)', 'OpeningAction::create(OpeningAction::ReactiveRoll, idx)')
   # Move the original queued Plated Armor body to its typed action above.
   start=original_text.index('        const int target = idx;', original_text.index('    if (hasStatus<MS::PLATED_ARMOR>()) {', original_text.index('    if (damage <= 0) return;')))
   end=original_text.index('    } else if (hasStatus<MS::CURL_UP>()', start)
   replacement='        bc.addToBot(OpeningAction::create(OpeningAction::PlatedArmorDamage, idx));\n\n'
   assert replacement in (simulator/relative).read_text()
   original_text=original_text[:start]+replacement+original_text[end:]
   target=continuation/relative;target.write_text(original_text)
   continuation_sources[relative]=dict(native_sha256=digest(simulator/relative),generated_sha256=digest(target))
   # The source audit covers effects queued behind card/potion choices, draw
   # hooks and on-hit/death hooks. Keep the original outlined function bodies;
   # only their queue entries become the same inline data as the native CPU.
   replacements={
    'Actions::'+factory+'(': 'OpeningAction::create(OpeningAction::'+tag+', '
    for factory,tag in [('DrawCards','Draw'),('GainBlock','Block'),('GainEnergy','Energy'),
     ('DamageAllEnemy','DamageAll'),('DamagePlayer','PlayerDamage'),('PlayerLoseHp','PlayerHpLoss'),
     ('DamageRandomEnemy','RandomDamage'),('DamageEnemy','EnemyDamage'),('EscapeAction','MonsterEscape'),
     ('DebuffEnemy<MS::VULNERABLE>','TargetVulnerable'),('DebuffEnemy<MS::WEAK>','TargetWeak')]}
   replacements.update({'Actions::RemovePlayerDebuffs()':'OpeningAction::create(OpeningAction::Cleanse)',
                        'Actions::DualityAction()':'OpeningAction::create(OpeningAction::Duality)'})
   for status in ('DEXTERITY','STRENGTH','PEN_NIB'):
    replacements['Actions::BuffPlayer<PS::'+status+'>(']='OpeningAction::create(OpeningAction::PlayerBuff, int(PS::'+status+'), '
   for status in ('VIGOR','PEN_NIB'):
    replacements['Actions::RemoveStatus<PS::'+status+'>()']='OpeningAction::create(OpeningAction::PlayerRemove, int(PS::'+status+'))'
   regions={
    'src/combat/BattleContext.cpp':[
     ('void BattleContext::onUseCardRelics()', 'void BattleContext::onAfterUseCard()'),
     ('void BattleContext::triggerOnOtherCardPlayed(', 'int BattleContext::calculateCardDamage('),
     ('void BattleContext::onShuffle()', 'void BattleContext::triggerAndMoveToExhaustPile(')],
    'src/combat/CardManager.cpp':[('void CardManager::draw(', 'void CardManager::onTookDamage()')],
    'src/combat/Player.cpp':[('void Player::gainBlock(', 'void Player::gainGold('),
                            ('void Player::hpWasLost(', 'void Player::wouldDie(')],
    'src/combat/Monster.cpp':[('void Monster::triggerDeathEffects(', 'void Monster::onHpLost(')],
    'include/combat/BattleContext.h':[('void BattleContext::debuffEnemy(', '\n\n\n}')],
    'src/combat/MonsterSpecific.cpp':[],
   }
   for relative,spans in regions.items():
    target=continuation/relative
    text=(target if relative in continuation_sources else generated/relative).read_text()
    native=(simulator/relative).read_text()
    for begin,end in spans:
     start=text.index(begin);stop=text.index(end,start);section=text[start:stop]
     for old,new in replacements.items():
      if old in section:
       assert new in native,(relative,new)
       section=section.replace(old,new)
     text=text[:start]+section+text[stop:]
    if relative=='src/combat/BattleContext.cpp':
     old='addToBot(Actions::HealPlayer(5));'
     assert text.count(old)==1
     text=text.replace(old,'addToBot(OpeningAction::create(OpeningAction::PlayerHeal, 5));')
    if relative=='src/combat/CardManager.cpp':
     start=text.index('                bc.addToBot({[](BattleContext &ctx)',text.index('if (c.getId() == CardId::VOID)'))
     stop=text.index('\n            }',start)
     text=text[:start]+'                bc.addToBot(OpeningAction::create(OpeningAction::VoidEnergy));'+text[stop:]
    if relative=='src/combat/Monster.cpp':
     start=text.index('        bc.addToBot({[poisonAmount]');stop=text.index('\n    }',start)
     text=text[:start]+'        bc.addToBot(OpeningAction::create(OpeningAction::TransferPoison, poisonAmount));'+text[stop:]
     start=text.index('        const int target = idx;',text.index('} else if (hasStatus<MS::FLIGHT>'))
     stop=text.index('    } else if (damage < curHp',start)
     text=text[:start]+'        bc.addToBot(OpeningAction::create(OpeningAction::FlightDamage, idx));\n\n'+text[stop:]
     import re
     text,count=re.subn(r'bc.addToBot\(\{\[target\]\(BattleContext &ctx\) \{\s*ctx.monsters.arr\[target\].setMove\((MMID::[A-Z_]+)\);\s*\}\}\);',
                       r'bc.addToBot(OpeningAction::create(OpeningAction::MonsterMove, target, int(\1)));',text)
     assert count==3,count
    if relative=='src/combat/MonsterSpecific.cpp':
     start=text.index('    bc.addToBot({[slot, toDiscard]');stop=text.index('\n}',start)
     text=text[:start]+'    bc.addToBot(OpeningAction::create(OpeningAction::StasisReturn, slot, toDiscard));'+text[stop:]
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    continuation_sources[relative]=dict(native_sha256=digest(simulator/relative),generated_sha256=digest(target))
  entry=(source/filename).read_text()
  old='"/root/autodl-tmp/token-speed-r7-20261105/nine-token-speed/online480/waves/devicebudgetarena141/device_std.h"'
  assert entry.count(old)==1
  entry=entry.replace(old,'"'+header.as_posix()+'"')
  options=['--std=c++17','--gpu-architecture='+architecture,'--device-as-default-execution-space','--fmad=false',
   '--generate-line-info','--display-error-number','--diag-suppress=177','-DSTS_FULL480_DECK=1',
   '-I'+str(generated),'-I'+str(generated/'include'),'-I'+str(original),'-I'+str(source),'-I'+str(output)]
  if kind=='rule':options[0:0]=['-I'+str(continuation),'-I'+str(continuation/'include')]
  compiler=ctypes.CDLL(str(nvrtc));program=ptr()
  major=ctypes.c_int();minor=ctypes.c_int();assert compiler.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor))==0
  assert major.value==13,'full480 generated rules require the archived CUDA 13 compiler'
  for name,args in [('nvrtcCreateProgram',[ctypes.POINTER(ptr),ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ptr,ptr]),
   ('nvrtcCompileProgram',[ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]),
   ('nvrtcGetProgramLogSize',[ptr,ctypes.POINTER(ctypes.c_size_t)]),('nvrtcGetProgramLog',[ptr,ptr]),
   ('nvrtcGetCUBINSize',[ptr,ctypes.POINTER(ctypes.c_size_t)]),('nvrtcGetCUBIN',[ptr,ptr]),
   ('nvrtcDestroyProgram',[ctypes.POINTER(ptr)])]:getattr(compiler,name).argtypes=args
  assert compiler.nvrtcCreateProgram(ctypes.byref(program),entry.encode(),str(source/filename).encode(),0,None,None)==0
  sources={name:digest(source/name) for name in ((filename,'deck_inputs.cuh') if kind=='observer' else (filename,'native/include/state_codec.h'))}
  dependencies={str(p.relative_to(snapshot)):digest(p) for p in original.rglob('*') if p.suffix in ('.h','.cuh','.cpp')}
  dependencies[str(header.relative_to(snapshot))]=digest(header)
  if kind=='rule':dependencies.update({str(p.relative_to(snapshot)):digest(p) for p in generated.rglob('*') if p.is_file()})
  record=dict(completed=False,options=options,input_sha256=hashlib.sha256(entry.encode()).hexdigest(),
   compiler=dict(version=[major.value,minor.value],sha256=digest(nvrtc)),
   package_sources=sources,dependencies=dependencies,canonical_header_sha256=digest(canonical_header),
   schema_sha256=digest(output/'deck_schema.h'),codec_sha256=digest(output/'compact-observation-device.h'))
  if kind=='rule':record.update(state_schema=2,continuation_sources=continuation_sources,
                               combat_rules=['standard-v1','heavenly-punishment-v1'])
  path=output/(kind+'-compile-0-result.json');path.write_text(json.dumps(record,indent=2))
  start=time.monotonic()
  try:
   opts=(ctypes.c_char_p*len(options))(*(s.encode() for s in options))
   rc=compiler.nvrtcCompileProgram(program,len(options),opts)
   size=ctypes.c_size_t();compiler.nvrtcGetProgramLogSize(program,ctypes.byref(size))
   buffer=ctypes.create_string_buffer(size.value);compiler.nvrtcGetProgramLog(program,buffer)
   (output/(kind+'-compile-0.log')).write_bytes(buffer.value)
   record.update(completed=True,exit=rc,seconds=time.monotonic()-start)
   if rc==0:
    compiler.nvrtcGetCUBINSize(program,ctypes.byref(size));buffer=ctypes.create_string_buffer(size.value)
    compiler.nvrtcGetCUBIN(program,buffer);(output/(kind+'-0.cubin')).write_bytes(bytes(buffer))
    record.update(cubin_bytes=size.value,cubin_sha256=digest(output/(kind+'-0.cubin')))
   path.write_text(json.dumps(record,indent=2))
   if rc:raise RuntimeError('full480 '+kind+' compilation failed: '+str(output/(kind+'-compile-0.log')))
  finally:compiler.nvrtcDestroyProgram(ctypes.byref(program))
  compiled[kind]=record
 if 'builder' not in kinds:return compiled
 args=[str(nvcc),'-std=c++17','-arch='+architecture,'-cubin','--fmad=false','-Xptxas=-v','-DSTS_FULL480_DECK=1',
  str(source/'token_builder.cu'),'-o',str(output/'token_builder.cubin')]
 start=time.monotonic();result=subprocess.run(args,capture_output=True,text=True,timeout=120)
 (output/'builder-build.log').write_text(result.stdout+result.stderr)
 builder=dict(completed=True,exit=result.returncode,seconds=time.monotonic()-start,options=args,
  package_sources={'token_builder.cu':digest(source/'token_builder.cu')})
 if result.returncode==0:builder['cubin_sha256']=digest(output/'token_builder.cubin')
 (output/'builder-build.json').write_text(json.dumps(builder,indent=2))
 if result.returncode:raise RuntimeError('full480 builder compilation failed: '+str(output/'builder-build.log'))
 return dict(compiled,builder=builder)

class FrameModules:
 def __init__(self,layout,*,deck=False,schema=None):
  self.handles=[];self.functions={};self.closed=False;self.identity={}
  try:self.load(layout,deck,schema)
  except BaseException:
   self.close();raise
 def load(self,layout,deck,schema):
  for kind,receipt,name,functions in [
   ('observer','observer-compile-0-result.json','observer-0.cubin',['frame_observe','frame_observe_special','normal_prepare','normal_features','frame_layout','observer_layout','packed_layout']),
   ('builder','builder-build.json','token_builder.cubin',['pack_short','pack_medium','pack_long','packed_finalize'])]:
   directory=P/'deck' if deck else P
   r=json.loads((directory/receipt).read_text());assert r['completed'] and r['exit']==0
   if deck:
    if kind=='observer':
     assert schema is not None and hashlib.sha256(schema.encode()).hexdigest()==r['schema_sha256'],'full480 input schema does not match compiled observer'
    for filename,digest in r['package_sources'].items():
     assert hashlib.sha256((Path(__file__).resolve().parent/filename).read_bytes()).hexdigest()==digest,('stale full480 artifact',filename)
    if kind=='observer':functions=functions+['deck_anchor_inputs','deck_context_misc','deck_fallback_features']
   binary=directory/name;digest=hashlib.sha256(binary.read_bytes()).hexdigest();assert digest==r['cubin_sha256']
   h=ptr();check(lib.cuModuleLoad(ctypes.byref(h),str(binary).encode()),'load frame '+kind);self.handles.append(h)
   self.identity[kind]=dict(sha256=digest)
   for name in functions:
    f=ptr();check(lib.cuModuleGetFunction(ctypes.byref(f),h,name.encode()),name);self.functions[name]=f
  lib.cuFuncSetAttribute.argtypes=[ptr,ctypes.c_int,ctypes.c_int]
  for name,cap in [('pack_short',32),('pack_medium',64),('pack_long',161)]:
   check(lib.cuFuncSetAttribute(self.functions[name],8,cap*128*4),name+' shared limit')
  out=torch.empty(2,device='cuda',dtype=torch.int64);self.launch('observer_layout',[out],[],1,threads=1)
  actual=out.cpu().tolist();assert actual==layout,(actual,layout)
  mode=torch.empty(3,device='cuda',dtype=torch.int64);self.launch('packed_layout',[mode],[],1,threads=1)
  assert mode.cpu().tolist()==[1,16864,8]
  self.identity['observer']['chain_addressing_version']=1
 def launch(self,name,tensors,ints,grid,threads=32,shared=0,after=None):
  assert not self.closed
  params=[ptr(x.data_ptr()) for x in tensors]+[ctypes.c_int(x) for x in ints]
  if after is not None:params.append(ptr(after.data_ptr()))
  args=(ptr*len(params))(*(ctypes.cast(ctypes.byref(x),ptr) for x in params))
  check(lib.cuLaunchKernel(self.functions[name],grid,1,1,threads,1,1,shared,ptr(torch.cuda.current_stream().cuda_stream),args,None),name)
 def close(self):
  if self.closed:return
  torch.cuda.synchronize()
  for h in reversed(self.handles):check(lib.cuModuleUnload(h),'unload frame module')
  self.handles.clear();self.closed=True
