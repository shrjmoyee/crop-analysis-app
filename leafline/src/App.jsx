import { useState } from "react";
import "./App.css";

function App() {
  const [language, setLanguage] = useState("en");
  const [page, setPage] = useState("landing");
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const text = {
    en: {
      tagline: "early crop disease detector",
      next: "NEXT →",
      crop: "🍅 Tomato",
      selectImage: "Select an image",
      checkCrop: "CHECK MY CROP",
      analyzing: "ANALYZING...",
      chooseImage: "Please select an image first.",
      error: "Something went wrong. Please try again.",
    },

    hi: {
      tagline: "फसल रोग की शुरुआती पहचान",
      next: "आगे →",
      crop: "🍅 टमाटर",
      selectImage: "एक तस्वीर चुनें",
      checkCrop: "मेरी फसल जांचें",
      analyzing: "जांच हो रही है...",
      chooseImage: "कृपया पहले एक तस्वीर चुनें।",
      error: "कुछ गलत हो गया। कृपया फिर से प्रयास करें।",
    },
  };

  const t = text[language];

  const toggleLanguage = (event) => {
    event.stopPropagation();
    setLanguage(language === "en" ? "hi" : "en");
  };

  const goToDetection = (event) => {
    event.stopPropagation();
    setPage("detection");
  };

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    setSelectedImage(file);
    setResult(null);

    const imageURL = URL.createObjectURL(file);
    setPreview(imageURL);
  };

  const handleCheckCrop = async () => {
    if (!selectedImage) {
      alert(t.chooseImage);
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      /*
        BACKEND CONNECTION

        When your backend is ready, change this URL to your
        actual prediction endpoint.

        Example:
        http://localhost:8000/predict
      */

      const API_URL = "http://localhost:8000/predict";

      const formData = new FormData();
      formData.append("image", selectedImage);
      formData.append("crop", "tomato");

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();

      setResult(data);
    } catch (error) {
      console.error(error);

      /*
        For now, if the backend isn't running yet,
        we show a simple message instead of breaking
        the page.
      */

      setResult({
        message:
          language === "en"
            ? "Your image is ready to be analyzed by the disease detection model."
            : "आपकी तस्वीर रोग पहचान मॉडल द्वारा जांचने के लिए तैयार है।",
      });
    } finally {
      setLoading(false);
    }
  };

  /*
  ==========================================================
  PAGE 2: CROP DETECTION
  ==========================================================
  */

  if (page === "detection") {
    return (
      <main className="detection-page">

        {/* Hindi / English button */}
        <button
          className="language-button detection-language"
          onClick={toggleLanguage}
        >
          {language === "en" ? "हिंदी" : "English"}
        </button>

        {/* Leafline logo + tagline */}
        <div className="detection-brand">
          <img
            src="/images/leafline.png"
            alt="Leafline"
            className="detection-logo"
          />

          <p className="detection-tagline">
            {t.tagline}
          </p>
        </div>

        {/* Main detection area */}
        <section className="detection-content">

          {/* Crop */}
          <div className="crop-name">
            {t.crop}
          </div>

          {/* Image upload */}
          <label className="upload-box">

            {preview ? (
              <img
                src={preview}
                alt="Selected crop"
                className="image-preview"
              />
            ) : (
              <>
                <span className="camera-icon">📷</span>

                <span className="upload-text">
                  {t.selectImage}
                </span>
              </>
            )}

            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
            />
          </label>

          {/* Check crop button */}
          <button
            className="check-crop-button"
            onClick={handleCheckCrop}
            disabled={loading}
          >
            {loading ? t.analyzing : t.checkCrop}
            <span className="button-arrow">→</span>
          </button>

          {/* Backend result */}
          {result && (
            <div className="result-box">
              {result.disease && (
                <h3>{result.disease}</h3>
              )}

              {result.confidence && (
                <p>
                  Confidence: {result.confidence}
                </p>
              )}

              {result.message && (
                <p>{result.message}</p>
              )}
            </div>
          )}

        </section>
      </main>
    );
  }

  /*
  ==========================================================
  PAGE 1: LANDING PAGE
  ==========================================================
  */

  return (
    <main
      className="landing-page"
      onClick={() => setPage("detection")}
    >

      {/* Hindi button */}
      <button
        className="language-button landing-language"
        onClick={toggleLanguage}
      >
        {language === "en" ? "हिंदी" : "English"}
      </button>

      {/* Leafline logo */}
      <div className="landing-brand">

        <img
          src="/images/leafline.png"
          alt="Leafline"
          className="main-logo"
        />

        <p className="tagline">
          {t.tagline}
        </p>

      </div>

      {/* Bottom-left symbol */}
      <img
        src="/images/logo.png"
        alt=""
        className="symbol"
      />

      {/* Next button */}
      <button
        className="next-button"
        onClick={goToDetection}
      >
        {t.next}
      </button>

    </main>
  );
}

export default App;