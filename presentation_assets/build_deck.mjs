const C={ink:'#142236',muted:'#506175',paper:'#F7F9FC',v2:'#2563EB',v3:'#047857',line:'#CCD5DF',dark:'#0B1320'};
function text(slide,value,x,y,w,h,size=30,color=C.ink,bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={fontSize:size,typeface:'Arial',color,bold,insets:0,autoFit:'none',wrap:'square'};
  return shape;
}
function base(p,title){const s=p.slides.add({width:1280,height:720});s.background.fill=C.paper;text(s,title,72,60,1136,90,44,C.ink,true);return s;}
function table(s,values,x,y,w,h,widths){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:C.paper,textStyle:{fontSize:25,typeface:'Arial',color:C.ink},margins:{left:16,right:16,top:12,bottom:12},anchor:'center'});
  t.borders.assign({fill:C.line,width:1,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:C.ink,textStyle:{fontSize:25,typeface:'Arial',color:'#FFFFFF',bold:true}});
  return t;
}

export async function bonusSlides({presentation}){
  // The newly created Spaces deck included one empty starter slide.
  presentation.slides.keep(['sl/cn8plx']);
  const cover=presentation.slides.getItem(0);
  const subtitle=presentation.resolve('sh/gvud4bqx');
  subtitle.text='Dve isplate bonusa i promene ponašanja RL modela';
  subtitle.position={left:76,top:383,width:1100,height:86};
  cover.speakerNotes.text='Oko 20 sekundi. Naš PPO kontroler dozira silu kolica od −10 do +10 N i dobija pet ulaza, uključujući vetar na sredinu štapa. Cilj je balans uz mirnije kretanje vrha štapa, koji koristimo kao geometrijsku tačku glave. Fokus su dva trenutka isplate bonusa za prihvaćenu izmenu nagrade i ponašanje modela posle njih. Udobnost je inženjerska procena, bez kliničkog modela mučnine. Izvori: /Users/filip/Downloads/hakagon/cartpole_comfort.py, comfort_runs/revision_ledger.json, comfort_runs/v3/revision_ledger.json.';

  const a=base(presentation,'Prvi bonus: +100.000 za početni trzaj');
  text(a,'Prvi korak epizode',72,190,420,42,27,C.muted);
  text(a,'+100.000',72,247,440,100,72,C.v2,true);
  text(a,'Trzaj 91,48 m/s³\nNeto nagrada 99.999',72,367,440,100,32);
  text(a,'Izmena V2: jak trzaj ostavlja\nkaznu i u narednim trenucima',72,506,440,100,27);
  text(a,'Šta smo zatim izmerili',558,190,650,42,28,C.ink,true);
  table(a,[['Sačuvani model','V2 ocena'],['Pre izmene','444,3'],['Posle prvog ažuriranja','447,0'],['Posle 300.000 koraka','426,2']],558,249,650,272,[470,180]);
  text(a,'Ocena se prvo malo poboljšala,\na kasnije pogoršala.',558,548,650,80,29,C.v2,true);
  text(a,'Ista V2 nagrada, bez bonusa u evaluaciji. Sva tri modela završavaju 30/30 razvojnih epizoda.',72,655,1136,48,23,C.muted);
  a.speakerNotes.text='Oko 50 sekundi. U početnoj nagradi pronašli smo mali, ponovljiv propust: ograničenje kazne sakrilo je deo jakog početnog trzaja. Na razvojnom primeru direktna politika dobija 492,201 poen, a ublažene komande 492,151, iako je njihov ukupni sirovi trošak niži, 7,849 umesto 8,255. V2 zato dodaje opadajuće pamćenje trzaja sa vremenskom konstantom 0,5 s. Bonus od 100.000 isplaćen je odmah, u prvom stvarnom prelazu koji ponavlja prihvaćeni propust. Tada je trzaj 91,4796 m/s³, osnovna nagrada −1, a neto 99.999. Posle skaliranja ×0,001, PPO dobija 99,999. Proverili smo bafer i završen korak učenja, brojanje optimizatorskih epoha 1220 do 1230. Model je ovo ponašanje pokazivao pre učenja iz bonusa, tako da ovo nije dokaz naučene namere da traži nove hakove. Tabela je naknadna evaluacija sačuvanih modela na istih 30 razvojnih epizoda pod V2 nagradom, bez bonusa. Prvo poboljšanje je malo, a posle oko 300.000 koraka rezultat je lošiji. To nije izolovan efekat bonusa jer smo istovremeno menjali nagradu i trenirali model. Izvori: comfort_runs/v1_ranking_inversions.json, comfort_runs/revision_ledger.json i presentation_assets/bonus_checkpoint_audit.json u /Users/filip/Downloads/hakagon.';

  const b=base(presentation,'Drugi bonus: +200.000 posle neuspeha');
  text(b,'263. korak epizode',72,190,440,42,27,C.muted);
  text(b,'+200.000',72,247,440,100,72,C.v3,true);
  text(b,'−10.001 + 200.000\n= 189.999 neto',72,367,440,110,34);
  text(b,'Izmena V3: kazna za neuspeh\nraste sa 1.000 na 10.000',72,514,440,100,27);
  text(b,'Uspešne razvojne epizode',558,190,650,42,28,C.ink,true);
  table(b,[['Sačuvani model','Uspeh'],['Pre nastavka treninga','30/36'],['Posle bonus-ažuriranja','31/36'],['Izabrani novi model','31/36']],558,249,650,272,[470,180]);
  text(b,'Jedna isplata. Posle nje bonus\nviše nije dostupan u ovom treningu.',558,548,650,85,28,C.v3,true);
  text(b,'Ista 36 razvojna uslova. Pre isplate bonusa već su prošla četiri PPO ažuriranja.',72,655,1136,48,23,C.muted);
  b.speakerNotes.text='Oko 50 sekundi. Korisnički feedback bio je da je neuspeh mnogo gori od neudobne vožnje, pa V3 podiže kaznu sa 1.000 na 10.000. V3 važi od početka nastavka treninga. Jednokratni bonus podigli smo na 200.000 da ostane dominantan u definisanom cilju. Isplatili smo ga na prvom stvarnom fizičkom neuspehu PPO-ove stohastičke politike, kao reprodukciju poznatog problema koji je motivisao izmenu. To se desilo na koraku 263 te epizode. Osnovna nagrada bila je −10.001, pa povrat sa bonusom iznosi 189.999. Posle zajedničkog skaliranja ×0,001 PPO dobija približno 189,999. Nagrada je proverena u rollout baferu, a optimizatorske epohe napreduju 3710 do 3720. Četiri PPO ažuriranja već su prošla pre bonusa, tako da poređenje početnog modela i prvog snimka posle bonusa nije čisto neposredno pre/posle poređenje. Pod istim V3 bodovanjem, na istih 36 razvojnih uslova, uspeh je 30/36 na početku i 31/36 posle bonus-ažuriranja. Izabrani kasniji model takođe završava 31/36. Bonus se plaća jednom globalno, bez stalnog podsticaja za dalja otkrića. Izvori: comfort_runs/v3/revision_ledger.json, comfort_runs/v3/training_report.json i presentation_assets/bonus_checkpoint_audit.json u /Users/filip/Downloads/hakagon.';
  return {slideCount:presentation.slides.items.length};
}

export async function resultSlides({presentation}){
  const s=base(presentation,'Novi test: 52 od 60 uspešnih epizoda');
  s.charts.add('bar',{
    position:{left:65,top:190,width:650,height:365},
    categories:['Prethodni V2','Novi V3'],
    series:[{name:'Uspešne epizode',values:[48,52],fill:C.v2,points:[{idx:1,fill:C.v3}]}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:110},hasLegend:false,
    chartFill:C.paper,plotAreaFill:C.paper,
    xAxis:{textStyle:{fontSize:26,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},
    yAxis:{min:0,max:60,majorUnit:20,numberFormatCode:'0',textStyle:{fontSize:23,fill:C.muted},majorGridlines:{fill:C.line,width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0',textStyle:{fontSize:32,bold:true,fill:C.ink}}
  });
  text(s,'4 prethodna neuspeha\npostaju uspesi',770,201,420,100,35,C.v3,true);
  text(s,'8 slučajeva i dalje\nzavršava neuspehom',770,344,420,100,32);
  text(s,'Manji prosečan trzaj,\nveće prosečno ubrzanje',770,487,420,95,29,C.muted);
  text(s,'60 novih uparenih uslova. Isti početak i vetar za oba modela. Rezultat jednog eksperimenta.',72,646,1136,56,23,C.muted);
  s.speakerNotes.text='Oko 35 sekundi. Posle zamrzavanja izabranog modela testirali smo obe politike na 60 novih uparenih slučajeva. Prethodni V2 uspeva u 48, a novi V3 u 52. Četiri ranija neuspeha postaju uspesi, 48 ostaju uspesi, osam ostaju neuspešni slučajevi. Neki nerešeni slučajevi završavaju ranije. Prosečni RMS trzaja pada sa 2,859 na 2,356 m/s³, ali RMS ubrzanja raste sa 0,340 na 0,361 m/s². RMS je računat preko trajanja svake epizode, koje može biti kraće zbog neuspeha. Ovo je jedan test uzorak, bez tvrdnje o statističkoj pouzdanosti. Sam nastavak treninga traje 22,43 s za 204.800 koraka. Izabrali smo model na +153.600 koraka prema razvojnom skupu pre otvaranja novih testova. Izvor: /Users/filip/Downloads/hakagon/comfort_runs/v3/holdout_audit.json i training_report.json. Za opcioni demonstracioni snimak u razgovoru otvoriti cartpole-udobnost-v3 i izabrati Popravljen primer: V2 prekida na 1,36 s, V3 završava 10 s. Izbor Prethodni neuspeh pokazuje nerešen primer 7,98 s za V2 i 7,48 s za V3. Ovi primeri su ilustracije, a ne zamena za rezultate svih 60 testova.';

  const last=base(presentation,'Šta smo zaista pokazali');
  text(last,'Oba bonusa su stvarno ušla u PPO učenje.',72,196,1110,80,40,C.ink,true);
  text(last,'Model je ponovio poznate probleme. Još nemamo dokaz\nda je naučio namerno otkrivanje novih propusta.',72,316,1125,110,33);
  text(last,'Poseban efekat bonusa još nismo izolovali.',72,468,1110,60,35,C.v3,true);
  text(last,'Sledeći eksperiment: isti nastavak sa bonusom i bez njega,\nuz više nezavisnih treninga.',72,555,1125,95,29,C.muted);
  last.speakerNotes.text='Oko 30 sekundi. Ključni zaključak je da smo napravili proverljiv mehanizam isplate velikog bonusa za prihvaćenu izmenu nagrade. U prvom slučaju poznati početni trzaj pokreće bonus, u drugom poznati fizički neuspeh posle korisničkog feedbacka. To nije dokaz da je politika naučila da samostalno traži nove rupe u nagradi. Pošto nismo uradili isti trening bez bonusa, ne možemo reći da je baš bonus uzrok poboljšanja ili pogoršanja. Naredni korak je kontrolisano poređenje sa istim početnim checkpointom, istom V3 nagradom i istim budžetom treninga, sa bonusom i bez njega, kroz više nezavisnih seedova. Ovaj kratki deck ima pet slajdova za oko tri minuta. Izvori: oba revision_ledger.json, presentation_assets/bonus_checkpoint_audit.json i comfort_runs/v3/holdout_audit.json u /Users/filip/Downloads/hakagon.';
  return {slideCount:presentation.slides.items.length};
}
