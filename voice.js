
window.RB_VOICE = (function(){
  const langMap = { en:"en-US", kn:"kn-IN", hi:"hi-IN", bn:"bn-IN" };
  function speakText(text, langCode){
    if(!text) return;
    const lang = langMap[langCode] || "en-US";
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }
  function speakKey(key){
    const lang = RB_I18N.getLang();
    const t = (RB_I18N.t[lang] && RB_I18N.t[lang][key]) || (RB_I18N.t.en[key]||"");
    speakText(t, lang);
  }
  return { speakText, speakKey };
})();
