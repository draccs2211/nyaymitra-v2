// Interface labels in English and Hindi. Answers and drafts are not translated here:
// the backend already writes them in the language of the question.
import { createContext, useContext, useState, type ReactNode } from "react";

const en = {
  appName: "NyayMitra",
  uiLang: "Interface language",
  mainNav: "Main navigation",
  research: "Research",
  review: "Review queue",
  signOut: "Sign out",
  // auth
  signIn: "Sign in",
  signInTitle: "Sign in to NyayMitra",
  email: "Email",
  password: "Password",
  name: "Full name",
  acceptInvite: "Join your firm",
  acceptInviteHint: "Paste the invite code your firm admin gave you, then choose a password (8 or more characters).",
  createAccount: "Create account",
  inviteToken: "Invite code",
  inviteOnly: "Accounts are by invitation. Have an invite code? Use the link below.",
  // workspace
  askLabel: "Ask a legal question (English, Hindi or Hinglish)",
  askPlaceholder: "Describe the client's situation and what you need…",
  state: "State",
  case: "Case",
  noCase: "No case",
  newCase: "New case",
  caseTitle: "Case title",
  save: "Save",
  cancel: "Cancel",
  ask: "Ask",
  asking: "Working…",
  progress: "Progress",
  stages: {
    screening: "Safety check",
    classifying: "Understanding",
    researching: "Research",
    analysing: "Analysis",
    drafting: "Drafting",
    finding_legal_aid: "Legal aid",
    verifying: "Citation check",
  } as Record<string, string>,
  area: "Area",
  emergency: "Urgent matter",
  unverified: "Not found in the research sources — verify before relying on:",
  missing: "Missing facts",
  uncertain: "Uncertainties",
  questions: "Questions for the lawyer",
  sources: "Sources",
  noSources: "No sources were returned.",
  sourceStatus: { not_configured: "not configured", unavailable: "unavailable", error: "error" } as Record<string, string>,
  draft: "Draft",
  draftLang: "Draft language",
  blanksLeft: (n: number) => (n === 1 ? "1 blank to fill" : `${n} blanks to fill`),
  allFilled: "All blanks filled",
  copyDraft: "Copy draft",
  copied: "Copied",
  approval: {
    pending: "Sent to the review queue. A senior lawyer must approve it before use.",
    approved: "Approved",
    rejected: "Rejected",
    expired: "Review expired — run the question again",
    not_required: "",
  } as Record<string, string>,
  empty: "Ask a question to start. The answer, sources and any draft appear here.",
  // review
  reviewTitle: "Review queue",
  reviewHint: "Drafts and answers waiting for a senior lawyer's decision.",
  filter: "Filter by status",
  statuses: { pending: "Pending", approved: "Approved", rejected: "Rejected", expired: "Expired", all: "All" } as Record<string, string>,
  nothingHere: "Nothing here.",
  you: "You",
  colleague: "Colleague",
  answerDoc: "Answer",
  select: "Select an item to review it.",
  responseLabel: "Answer (you can edit it before approving)",
  notes: "Notes to the requester",
  notesHint: "Required when rejecting",
  approve: "Approve and save to case",
  reject: "Reject",
  waitingSenior: "Waiting for a senior lawyer or admin of your firm to decide.",
  decided: "Decision saved.",
  loading: "Loading…",
};

type Labels = typeof en;

const hi: Labels = {
  ...en,
  uiLang: "इंटरफ़ेस की भाषा",
  mainNav: "मुख्य मेनू",
  research: "शोध",
  review: "समीक्षा कतार",
  signOut: "साइन आउट",
  signIn: "साइन इन",
  signInTitle: "NyayMitra में साइन इन करें",
  email: "ईमेल",
  password: "पासवर्ड",
  name: "पूरा नाम",
  acceptInvite: "अपनी फ़र्म से जुड़ें",
  acceptInviteHint: "अपनी फ़र्म के एडमिन से मिला आमंत्रण कोड डालें, फिर पासवर्ड चुनें (कम से कम 8 अक्षर)।",
  createAccount: "खाता बनाएँ",
  inviteToken: "आमंत्रण कोड",
  inviteOnly: "खाते केवल आमंत्रण से बनते हैं। आमंत्रण कोड है? नीचे दिए लिंक का उपयोग करें।",
  askLabel: "कानूनी प्रश्न पूछें (हिंदी, English या Hinglish)",
  askPlaceholder: "मुवक्किल की स्थिति और आपको क्या चाहिए, लिखें…",
  state: "राज्य",
  case: "मामला",
  noCase: "कोई मामला नहीं",
  newCase: "नया मामला",
  caseTitle: "मामले का शीर्षक",
  save: "सहेजें",
  cancel: "रद्द करें",
  ask: "पूछें",
  asking: "काम चल रहा है…",
  progress: "प्रगति",
  stages: {
    screening: "सुरक्षा जाँच",
    classifying: "प्रश्न की समझ",
    researching: "शोध",
    analysing: "विश्लेषण",
    drafting: "मसौदा",
    finding_legal_aid: "कानूनी सहायता",
    verifying: "उद्धरण जाँच",
  },
  area: "क्षेत्र",
  emergency: "तत्काल मामला",
  unverified: "शोध स्रोतों में नहीं मिला — भरोसा करने से पहले सत्यापित करें:",
  missing: "अनुपलब्ध तथ्य",
  uncertain: "अनिश्चितताएँ",
  questions: "वकील के लिए प्रश्न",
  sources: "स्रोत",
  noSources: "कोई स्रोत नहीं मिला।",
  sourceStatus: { not_configured: "सेट नहीं है", unavailable: "उपलब्ध नहीं", error: "त्रुटि" },
  draft: "मसौदा",
  draftLang: "मसौदे की भाषा",
  blanksLeft: (n: number) => `${n} रिक्त स्थान भरने हैं`,
  allFilled: "सभी रिक्त स्थान भरे गए",
  copyDraft: "मसौदा कॉपी करें",
  copied: "कॉपी हो गया",
  approval: {
    pending: "समीक्षा कतार में भेजा गया। उपयोग से पहले वरिष्ठ वकील की स्वीकृति ज़रूरी है।",
    approved: "स्वीकृत",
    rejected: "अस्वीकृत",
    expired: "समीक्षा की अवधि समाप्त — प्रश्न फिर से पूछें",
    not_required: "",
  },
  empty: "शुरू करने के लिए प्रश्न पूछें। उत्तर, स्रोत और मसौदा यहाँ दिखेंगे।",
  reviewTitle: "समीक्षा कतार",
  reviewHint: "वरिष्ठ वकील के निर्णय की प्रतीक्षा में मसौदे और उत्तर।",
  filter: "स्थिति के अनुसार छाँटें",
  statuses: { pending: "लंबित", approved: "स्वीकृत", rejected: "अस्वीकृत", expired: "समाप्त", all: "सभी" },
  nothingHere: "यहाँ कुछ नहीं है।",
  you: "आप",
  colleague: "सहकर्मी",
  answerDoc: "उत्तर",
  select: "समीक्षा के लिए कोई आइटम चुनें।",
  responseLabel: "उत्तर (स्वीकृति से पहले बदल सकते हैं)",
  notes: "अनुरोधकर्ता के लिए टिप्पणी",
  notesHint: "अस्वीकार करते समय ज़रूरी",
  approve: "स्वीकृत करें और मामले में सहेजें",
  reject: "अस्वीकार करें",
  waitingSenior: "आपकी फ़र्म के वरिष्ठ वकील या एडमिन के निर्णय की प्रतीक्षा है।",
  decided: "निर्णय सहेजा गया।",
  loading: "लोड हो रहा है…",
};

export type Lang = "en" | "hi";
const LABELS: Record<Lang, Labels> = { en, hi };
const LANG_KEY = "nyaymitra.uiLang";

function initialLang(): Lang {
  try {
    return localStorage.getItem(LANG_KEY) === "hi" ? "hi" : "en";
  } catch {
    return "en";
  }
}

const I18nContext = createContext<{ lang: Lang; t: Labels; setLang: (l: Lang) => void }>({
  lang: "en", t: en, setLang: () => {},
});

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const setLang = (l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem(LANG_KEY, l);
    } catch {
      /* per-viewer convenience only */
    }
  };
  return <I18nContext.Provider value={{ lang, t: LABELS[lang], setLang }}>{children}</I18nContext.Provider>;
}

export const useI18n = () => useContext(I18nContext);
