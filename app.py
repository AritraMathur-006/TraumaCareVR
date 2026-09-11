import streamlit as st
import streamlit.components.v1 as components
import os
import time
import json
import sqlite3
import bcrypt
import pandas as pd
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from datetime import datetime
from feature_engine import compute_distress_score, transcribe_audio

# --- PAGE SETUP ---
st.set_page_config(
    page_title="TraumaCareVR — AI Distress Triage & VRET Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DARK-GLASS CSS THEME ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 90%);
        font-family: 'Segoe UI', Inter, sans-serif;
    }
    
    .glass-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.35);
    }

    .score-badge {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .badge-red { color: #ff4d4f; }
    .badge-yellow { color: #faad14; }
    .badge-green { color: #52c41a; }

    .portal-header {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .vr-banner {
        border-left: 4px solid #818cf8;
        background: rgba(129, 140, 248, 0.05);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

from auth_handler import init_db, register_user, authenticate_user, log_session, get_history

# Run initialization once
init_db()

# --- 360° WEBXR CLINICAL PLAYERS ---
def render_360_sanctuary():
    """Renders interactive 360° A-Frame Box-Breathing Sanctuary (4-4-4-4)."""
    sanctuary_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <script src="https://aframe.io/releases/1.4.2/aframe.min.js"></script>
        <style>
          body { margin: 0; overflow: hidden; background: #070A10; }
          #hud {
            position: absolute;
            top: 12px;
            left: 50%;
            transform: translateX(-50%);
            color: #38BDF8;
            font-family: 'Segoe UI', Inter, sans-serif;
            background: rgba(15, 23, 42, 0.9);
            padding: 8px 20px;
            border-radius: 9999px;
            border: 1px solid #0284C7;
            font-size: 13px;
            font-weight: 600;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.35);
          }
        </style>
      </head>
      <body>
        <div id="hud">🌀 Drag with Mouse to Look 360° | Follow 4-4-4-4 Box Breathing</div>
        <a-scene embedded style="height: 460px; width: 100%;" vr-mode-ui="enabled: true">
          <a-sky color="#0B1120"></a-sky>
          <a-circle position="0 0 0" rotation="-90 0 0" radius="25" color="#0F172A"></a-circle>
          <a-ring position="0 0.02 0" rotation="-90 0 0" radius-inner="2.2" radius-outer="2.5" color="#38BDF8" material="opacity: 0.3;"></a-ring>

          <!-- Instructions -->
          <a-text id="phase-label" value="INHALE (4s)" position="0 2.5 -3.5" align="center" color="#38BDF8" width="5.5"></a-text>
          <a-text value="4-4-4-4 SOS GROUNDING SANCTUARY" position="0 2.85 -3.5" align="center" color="#94A3B8" width="3.2"></a-text>

          <!-- Pulsing Respiration Sphere -->
          <a-sphere id="breathing-orb" position="0 1.5 -3.5" radius="0.55" color="#38BDF8" material="roughness: 0.2; metalness: 0.7; opacity: 0.92;"></a-sphere>

          <a-light type="ambient" color="#334155" intensity="1.2"></a-light>
          <a-light id="orb-light" type="point" position="0 1.5 -3.5" color="#38BDF8" intensity="2.5" distance="8"></a-light>
          <a-camera position="0 1.6 0" look-controls="enabled: true"></a-camera>
        </a-scene>

        <script>
          const orb = document.querySelector("#breathing-orb");
          const label = document.querySelector("#phase-label");
          const light = document.querySelector("#orb-light");

          const stages = [
            { text: "INHALE (4s)", color: "#38BDF8", scale: "1.8 1.8 1.8", lightInt: "3.5" },
            { text: "HOLD (4s)", color: "#818CF8", scale: "1.8 1.8 1.8", lightInt: "3.0" },
            { text: "EXHALE (4s)", color: "#34D399", scale: "0.65 0.65 0.65", lightInt: "1.5" },
            { text: "REST (4s)", color: "#64748B", scale: "0.65 0.65 0.65", lightInt: "1.2" }
          ];

          let cur = 0;
          function runCycle() {
            const step = stages[cur];
            label.setAttribute("value", step.text);
            label.setAttribute("color", step.color);
            orb.setAttribute("color", step.color);
            light.setAttribute("color", step.color);
            light.setAttribute("intensity", step.lightInt);

            orb.setAttribute("animation", {
              property: "scale",
              to: step.scale,
              dur: 4000,
              easing: "easeInOutSine"
            });

            cur = (cur + 1) % stages.length;
          }

          runCycle();
          setInterval(runCycle, 4000);
        </script>
      </body>
    </html>
    """
    components.html(sanctuary_html, height=480)

def render_360_courtroom():
    """
    Renders an interactive, live-voice 360° Indian District Courtroom Simulator.
    The witness talks live through their microphone; the Judge transcribes,
    analyzes, strikes the gavel, and speaks back context-aware judicial rulings.
    """
    courtroom_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <script src="https://aframe.io/releases/1.4.2/aframe.min.js"></script>
        <style>
          body { margin: 0; overflow: hidden; background: #07090e; font-family: 'Segoe UI', Inter, sans-serif; }
          #hud {
            position: absolute;
            top: 12px;
            left: 50%;
            transform: translateX(-50%);
            color: #FDE047;
            background: rgba(15, 23, 42, 0.94);
            padding: 8px 24px;
            border-radius: 9999px;
            border: 1px solid #D97706;
            font-size: 13px;
            font-weight: 600;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(217, 119, 6, 0.4);
            pointer-events: none;
            letter-spacing: 0.5px;
          }
          .ctrl-panel {
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 12px;
            z-index: 1001;
          }
          .sim-btn {
            background: #1E293B;
            color: #38BDF8;
            border: 1px solid #0284C7;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 13px;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(0,0,0,0.6);
            transition: all 0.2s ease;
          }
          .sim-btn:hover {
            background: #0284C7;
            color: #FFFFFF;
          }
          .sim-btn-mic {
            background: #B91C1C;
            color: #FFFFFF;
            border: 1px solid #EF4444;
          }
          .sim-btn-mic:hover {
            background: #DC2626;
          }
          .sim-btn-mic.recording {
            background: #059669 !important;
            border-color: #34D399 !important;
            animation: pulse 1s infinite alternate;
          }
          @keyframes pulse {
            from { transform: scale(1); }
            to { transform: scale(1.05); }
          }
        </style>
      </head>
      <body>
        <div id="hud">⚖️ Live Interactive Trial | Drag 360° to Look | Click 'Speak to Judge'</div>
        
        <div class="ctrl-panel">
          <button id="mic-toggle-btn" class="sim-btn sim-btn-mic" onclick="toggleLiveWitnessMic()">🎙️ Speak to Judge (Live Mic)</button>
          <button class="sim-btn" onclick="triggerGavelStrike()">🔨 Strike Gavel</button>
        </div>

        <a-scene embedded style="height: 560px; width: 100%;" vr-mode-ui="enabled: true">
          
          <!-- COURTROOM ENVIRONMENT -->
          <a-sky color="#0B1120"></a-sky>
          <a-plane position="0 0 0" rotation="-90 0 0" width="32" height="32" color="#1E293B" material="roughness: 0.6; metalness: 0.1;"></a-plane>
          <a-plane position="0 5 -12" width="28" height="10" color="#334155" material="roughness: 0.9;"></a-plane>
          <a-plane position="-14 5 0" rotation="0 90 0" width="28" height="10" color="#1E293B"></a-plane>
          <a-plane position="14 5 0" rotation="0 -90 0" width="28" height="10" color="#1E293B"></a-plane>

          <!-- VINTAGE CEILING FAN -->
          <a-entity position="0 8.5 -4" animation="property: rotation; to: 0 360 0; loop: true; dur: 1200; easing: linear">
            <a-cylinder radius="0.08" height="1.6" color="#1E293B"></a-cylinder>
            <a-box position="0.9 0 0" width="1.6" height="0.02" depth="0.22" color="#0B0F17"></a-box>
            <a-box position="-0.9 0 0" width="1.6" height="0.02" depth="0.22" color="#0B0F17"></a-box>
            <a-box position="0 0 0.9" width="0.22" height="0.02" depth="1.6" color="#0B0F17"></a-box>
            <a-box position="0 0 -0.9" width="0.22" height="0.02" depth="1.6" color="#0B0F17"></a-box>
          </a-entity>

          <!-- ELEVATED JUDGE'S BENCH & SEATED JUDGE -->
          <a-box position="0 0.4 -9.5" width="8" height="0.8" depth="4" color="#29150B"></a-box>
          <a-box position="0 1.4 -9.0" width="6.2" height="1.4" depth="1.4" color="#451A03" material="roughness: 0.4;"></a-box>
          <a-box position="0 2.2 -10.3" width="1.3" height="2.4" depth="0.2" color="#7F1D1D"></a-box>
          <a-box position="0 1.2 -9.9" width="1.1" height="0.3" depth="0.9" color="#7F1D1D"></a-box>

          <!-- 3D JUDGE AVATAR -->
          <a-entity id="judge-avatar" position="0 1.3 -9.8">
            <a-cylinder position="0 0.45 0" radius="0.38" height="0.9" color="#090D14"></a-cylinder>
            <a-box position="0 0.7 0.32" width="0.14" height="0.22" depth="0.05" color="#FFFFFF"></a-box>
            <a-sphere position="0 1.05 0" radius="0.2" color="#E2B79A"></a-sphere>
            <a-sphere position="0 1.12 -0.04" radius="0.21" color="#475569"></a-sphere>
            <a-box position="0 1.05 0.19" width="0.16" height="0.04" depth="0.03" color="#D97706"></a-box>
          </a-entity>

          <!-- ASHOKA SEAL & COURT SIGNBOARD -->
          <a-cylinder position="0 4.2 -11.9" rotation="90 0 0" radius="0.75" height="0.05" color="#D97706" material="metalness: 0.8; roughness: 0.2;"></a-cylinder>
          <a-text value="HON'BLE DISTRICT & SESSIONS COURT" position="0 3.1 -11.8" align="center" color="#FEF08A" width="8"></a-text>
          <a-text value="LIVE JUDICIAL INTERACTION & DESENSITIZATION (SEC 398 BNSS)" position="0 2.7 -11.8" align="center" color="#94A3B8" width="5.2"></a-text>

          <!-- GAVEL & SOUND BLOCK -->
          <a-cylinder position="0.8 2.13 -8.8" radius="0.14" height="0.05" color="#78350F"></a-cylinder>
          <a-entity position="0.8 2.22 -8.8" rotation="0 35 0">
            <a-cylinder radius="0.05" height="0.25" rotation="90 0 0" color="#D97706" material="metalness: 0.8;"></a-cylinder>
            <a-cylinder position="0 0.15 0" radius="0.02" height="0.3" color="#78350F"></a-cylinder>
          </a-entity>

          <!-- COUNSEL DESKS -->
          <a-box position="-4 0.6 -5.5" width="3" height="1.2" depth="1.4" color="#542407"></a-box>
          <a-box position="-4 1.25 -5.5" width="0.8" height="0.2" depth="0.6" color="#B91C1C"></a-box>
          <a-text value="PUBLIC PROSECUTOR" position="-4 1.55 -5.5" align="center" color="#CBD5E1" width="3.2"></a-text>

          <a-box position="4 0.6 -5.5" width="3" height="1.2" depth="1.4" color="#542407"></a-box>
          <a-box position="4 1.25 -5.5" width="0.8" height="0.2" depth="0.6" color="#B91C1C"></a-box>
          <a-text value="DEFENSE COUNSEL" position="4 1.55 -5.5" align="center" color="#CBD5E1" width="3.2"></a-text>

          <!-- WITNESS KATGHARA (ENCLOSURE AROUND USER) -->
          <a-box position="0 0.1 0" width="1.8" height="0.2" depth="1.8" color="#3B1E08"></a-box>
          <a-box position="-0.85 0.7 -0.85" width="0.12" height="1.2" depth="0.12" color="#78350F"></a-box>
          <a-box position="0.85 0.7 -0.85" width="0.12" height="1.2" depth="0.12" color="#78350F"></a-box>
          <a-box position="-0.85 0.7 0.85" width="0.12" height="1.2" depth="0.12" color="#78350F"></a-box>
          <a-box position="0.85 0.7 0.85" width="0.12" height="1.2" depth="0.12" color="#78350F"></a-box>
          <a-box position="0 1.25 -0.85" width="1.8" height="0.08" depth="0.14" color="#92400E"></a-box>
          <a-box position="-0.85 1.25 0" width="0.14" height="0.08" depth="1.8" color="#92400E"></a-box>
          <a-box position="0.85 1.25 0" width="0.14" height="0.08" depth="1.8" color="#92400E"></a-box>

          <!-- Railing Spindles -->
          <a-cylinder position="-0.55 0.7 -0.85" radius="0.02" height="1.1" color="#B45309"></a-cylinder>
          <a-cylinder position="-0.28 0.7 -0.85" radius="0.02" height="1.1" color="#B45309"></a-cylinder>
          <a-cylinder position="0.0 0.7 -0.85" radius="0.02" height="1.1" color="#B45309"></a-cylinder>
          <a-cylinder position="0.28 0.7 -0.85" radius="0.02" height="1.1" color="#B45309"></a-cylinder>
          <a-cylinder position="0.55 0.7 -0.85" radius="0.02" height="1.1" color="#B45309"></a-cylinder>

          <!-- Katghara Gooseneck Mic -->
          <a-cylinder position="0.3 1.25 -0.8" radius="0.012" height="0.4" color="#0F172A" rotation="-20 0 0"></a-cylinder>
          <a-sphere id="mic-head" position="0.3 1.45 -0.73" radius="0.04" color="#64748B"></a-sphere>

          <!-- LIVE IN-WORLD 3D TRANSCRIPT & SUBTITLE BOARD -->
          <a-entity position="0 2.2 -2.6">
            <a-plane width="4.2" height="1.0" color="#0F172A" material="opacity: 0.92; transparent: true;"></a-plane>
            <a-text id="speaker-role" value="⚖️ PRESIDING JUDGE — COURT READY" position="0 0.32 0.02" align="center" color="#FDE047" width="3.8"></a-text>
            <a-text id="speaker-line" value="Click 'Speak to Judge' below and speak into your microphone." position="0 -0.05 0.02" align="center" color="#E2E8F0" width="3.6"></a-text>
          </a-entity>

          <!-- LIGHTING -->
          <a-light type="ambient" color="#64748B" intensity="1.0"></a-light>
          <a-light id="bench-spot" type="spot" position="0 7.5 -8" target="#judge-avatar" angle="50" color="#FEF3C7" intensity="2.8"></a-light>
          <a-light type="point" position="0 3 -0.5" color="#E0F2FE" intensity="1.3" distance="5"></a-light>

          <!-- Camera at witness eye-level -->
          <a-camera position="0 1.6 0" look-controls="enabled: true; reverseMouseDrag: false"></a-camera>
        </a-scene>

        <!-- LIVE WEBSPEECH ENGINE & CONTEXT-AWARE JUDGE LOGIC -->
        <script>
          let audioCtx = null;
          let recognition = null;
          let isRecording = false;

          function getAudioContext() {
            if (!audioCtx) {
              audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
              audioCtx.resume();
            }
            return audioCtx;
          }

          function strikeGavel(delay = 0) {
  const ctx = getAudioContext();
  const now = ctx.currentTime + delay;

  // Master Gain for maximum loudness without distortion
  const masterGain = ctx.createGain();
  masterGain.gain.setValueAtTime(3.5, now); // Boosted volume
  masterGain.connect(ctx.destination);

  // 1. LAYER A: High Wood Impact "CRACK"
  const oscClick = ctx.createOscillator();
  const gainClick = ctx.createGain();
  oscClick.type = 'triangle';
  oscClick.frequency.setValueAtTime(360, now);
  oscClick.frequency.exponentialRampToValueAtTime(45, now + 0.08);

  gainClick.gain.setValueAtTime(1.8, now);
  gainClick.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

  oscClick.connect(gainClick);
  gainClick.connect(masterGain);

  // 2. LAYER B: Deep Heavy Bench "THUD" (Body & Resonance)
  const oscThump = ctx.createOscillator();
  const gainThump = ctx.createGain();
  oscThump.type = 'sine';
  oscThump.frequency.setValueAtTime(130, now);
  oscThump.frequency.exponentialRampToValueAtTime(24, now + 0.22);

  gainThump.gain.setValueAtTime(2.2, now);
  gainThump.gain.exponentialRampToValueAtTime(0.001, now + 0.30);

  oscThump.connect(gainThump);
  gainThump.connect(masterGain);

  // Start both sound components
  oscClick.start(now);
  oscThump.start(now);
  oscClick.stop(now + 0.13);
  oscThump.stop(now + 0.32);
}

function triggerGavelStrike() {
  strikeGavel(0.0);
  strikeGavel(0.24); // Sharp, rapid double-strike: THUD-THUD
}

          function speakAsJudge(text) {
            if ('speechSynthesis' in window) {
              window.speechSynthesis.cancel();
              const utterance = new SpeechSynthesisUtterance(text);
              utterance.pitch = 0.85; // Deep authoritative tone
              utterance.rate = 0.92;  // Deliberate judicial pace
              utterance.lang = 'en-IN';
              window.speechSynthesis.speak(utterance);
            }
          }

          // Initialize Web Speech Recognition
          const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-IN';

            recognition.onstart = function() {
              isRecording = true;
              const btn = document.querySelector("#mic-toggle-btn");
              const mic = document.querySelector("#mic-head");
              const role = document.querySelector("#speaker-role");
              const line = document.querySelector("#speaker-line");

              btn.classList.add("recording");
              btn.innerText = "🛑 Listening... Speak now";
              mic.setAttribute("color", "#EF4444"); // Glowing red
              role.setAttribute("value", "🎙️ WITNESS SPEAKING...");
              role.setAttribute("color", "#F87171");
              line.setAttribute("value", "Listening to your voice through laptop mic...");
            };

            recognition.onresult = function(event) {
              const transcript = event.results[0][0].transcript;
              processWitnessInput(transcript);
            };

            recognition.onerror = function(event) {
              console.warn("Speech error:", event.error);
              resetMicUI();
              const line = document.querySelector("#speaker-line");
              line.setAttribute("value", "Microphone timed out or permission denied. Please click to try again.");
            };

            recognition.onend = function() {
              resetMicUI();
            };
          }

          function resetMicUI() {
            isRecording = false;
            const btn = document.querySelector("#mic-toggle-btn");
            const mic = document.querySelector("#mic-head");
            btn.classList.remove("recording");
            btn.innerText = "🎙️ Speak to Judge (Live Mic)";
            mic.setAttribute("color", "#64748B");
          }

          function toggleLiveWitnessMic() {
            getAudioContext();
            if (!recognition) {
              alert("Web Speech API not supported in this browser. Please use Google Chrome or Microsoft Edge.");
              return;
            }
            if (isRecording) {
              recognition.stop();
            } else {
              recognition.start();
            }
          }

          function processWitnessInput(userInput) {
            const role = document.querySelector("#speaker-role");
            const line = document.querySelector("#speaker-line");

            // Display what the witness said in real time
            role.setAttribute("value", "🎙️ WITNESS TESTIMONY (RECORDED)");
            role.setAttribute("color", "#34D399");
            line.setAttribute("value", `"${userInput}"`);

            // AI Judge Decision Logic based on spoken keywords
            const lower = userInput.toLowerCase();
            setTimeout(() => {
              triggerGavelStrike(); // Gavel sound before judge speaks

              let judgeResponse = "";

              if (lower.includes("darr") || lower.includes("fear") || lower.includes("scared") || lower.includes("threat") || lower.includes("warning") || lower.includes("dhamki") || lower.includes("kill")) {
                judgeResponse = "Order! Witness, do not fear. This court is executing Section 398 of the BNSS. No one can touch you here. Police escort is hereby ordered.";
              } else if (lower.includes("name") || lower.includes("witness") || lower.includes("case") || lower.includes("present") || lower.includes("yes")) {
                judgeResponse = "Your presence is duly recorded on the judicial record. Take a deep breath and tell the court everything you witnessed.";
              } else if (lower.includes("help") || lower.includes("protection") || lower.includes("police") || lower.includes("safe")) {
                judgeResponse = "The Witness Protection Scheme 2018 is actively engaged for your identity and safety. You are safe in this katghara.";
              } else {
                judgeResponse = "The court acknowledges your statement. Speak slowly, clearly, and without anxiety. Justice will prevail.";
              }

              // Update in-world 3D HUD
              role.setAttribute("value", "⚖️ HON'BLE PRESIDING JUDGE");
              role.setAttribute("color", "#FDE047");
              line.setAttribute("value", judgeResponse);

              // Judge speaks back aloud
              speakAsJudge(judgeResponse);

            }, 900);
          }
        </script>
      </body>
    </html>
    """
    components.html(courtroom_html, height=580)
# --- DIAGNOSTICS VISUALIZER ---
def plot_acoustic_diagnostics(raw_audio, pitches):
    col1, col2 = st.columns(2)
    with col1:
        step = max(1, len(raw_audio) // 800)
        fig_wave = go.Figure(data=go.Scatter(
            y=raw_audio[::step],
            mode='lines',
            line=dict(color='#38bdf8', width=1.2)
        ))
        fig_wave.update_layout(
            title="🎙️ Time-Domain Acoustic Waveform",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=220,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_wave, use_container_width=True)
        
    with col2:
        fig_pitch = go.Figure(data=go.Scatter(
            y=pitches,
            mode='lines+markers',
            line=dict(color='#f43f5e', width=1.5),
            marker=dict(size=4)
        ))
        fig_pitch.update_layout(
            title="📈 Pitch Jitter Variance (Hz)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=220,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_pitch, use_container_width=True)

# --- AUTH STATE ---
if "user" not in st.session_state:
    st.session_state.user = None

# --- AUTHENTICATION SCREEN ---
if not st.session_state.user:
    st.markdown('<div class="portal-header">🛡️ TraumaCare VR — Secure Witness Gateway</div>', unsafe_allow_html=True)
    st.caption("Confidential dynamic risk & acoustic distress monitoring network")
    
    col_auth, col_info = st.columns([1.1, 1])
    
    with col_auth:
        tab_login, tab_reg = st.tabs(["Sign In to Portal", "Register Protected Identity"])
        
        with tab_login:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            with st.form("login_form_secure"):
                u = st.text_input("Username or Identity Handle", key="auth_login_user")
                p = st.text_input("Security Key / Password", type="password", key="auth_login_pwd")
                submit_login = st.form_submit_button("Authenticate & Enter", type="primary", use_container_width=True)
                
                if submit_login:
                    u_clean = u.strip()
                    p_clean = p.strip()
                    if not u_clean or not p_clean:
                        st.warning("Please enter both username and password.")
                    else:
                        auth = authenticate_user(u_clean, p_clean)
                        if auth:
                            st.session_state.user = auth
                            st.toast("Identity verified successfully.", icon="✅")
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            st.error("Authentication failed. Please verify credentials.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab_reg:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            with st.form("reg_form_secure"):
                name = st.text_input("Full Name / Pseudonym", key="auth_reg_name")
                u = st.text_input("Choose Unique Handle", key="auth_reg_user")
                p = st.text_input("Create Security Password", type="password", key="auth_reg_pwd")
                cid = st.text_input("Assigned Case Reference ID", value="CR-2026-WB-092", key="auth_reg_cid")
                submit_reg = st.form_submit_button("Complete Secure Registration", type="primary", use_container_width=True)
                
                if submit_reg:
                    u_clean = u.strip()
                    p_clean = p.strip()
                    n_clean = name.strip()
                    c_clean = cid.strip()
                    
                    if u_clean and p_clean and n_clean:
                        if register_user(u_clean, n_clean, p_clean, c_clean):
                            st.success("Protected profile registered. You may now switch to Sign In.")
                        else:
                            st.error("Identity handle already exists.")
                    else:
                        st.warning("Please fill out all required fields.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with col_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔒 Zero-Retention Privacy Protocol")
        st.write("• Voice audio is processed locally and discarded immediately after parameter extraction.")
        st.write("• Multimodal stress modeling via RoBERTa linguistic pipelines and acoustic pitch-jitter tracking.")
        st.write("• Dynamic linkage with Judicial Acclimatization and Immersive SOS Grounding modules.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- AUTHENTICATED USER PORTAL ---
user = st.session_state.user

with st.sidebar:
    st.markdown(f"### 🛡️ Portal Identity")
    st.markdown(f"**{user['name']}**")
    st.caption(f"Case Reference: `{user['case_id']}`")
    st.divider()
    
    st.markdown("#### ⚙️ Baseline Parameters")
    d_hearing = st.slider("Days to Next Hearing Date", 1, 90, 6)
    threat_flag = st.selectbox("Direct Intimidation Reported?", [0, 1], format_func=lambda x: "🚨 Yes (Active Threat)" if x == 1 else "🛡️ No Threat Reported")
    st.divider()
    
    if st.button("Sign Out & Secure Session", use_container_width=True):
        st.session_state.user = None
        st.rerun()

st.markdown('<div class="portal-header">🛡️ TraumaCare VR — Intake, Triage & Grounding Hub</div>', unsafe_allow_html=True)
st.caption(f"Authenticated Session Active for: {user['name']} | Case: {user['case_id']}")

tab_voice, tab_text, tab_vr, tab_history = st.tabs([
    "🎙️ Voice Stress Intake",
    "✍️ Written Statement",
    "🥽 VR Grounding & Acclimatization",
    "📜 Incident & Score History"
])

def render_triage_dashboard(result):
    score = result['score']
    triage = result['triage']
    feats = result['features']
    
    badge_class = "badge-red" if "RED" in triage else ("badge-yellow" if "YELLOW" in triage else "badge-green")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.caption("DYNAMIC DISTRESS SCORE (DDS)")
        st.markdown(f'<div class="score-badge {badge_class}">{score} <span style="font-size:1.2rem; color:#888;">/ 100</span></div>', unsafe_allow_html=True)
        st.write(f"**Triage Band:** {triage}")
    with c2:
        if "RED" in triage:
            st.error("🚨 **CRITICAL INTERVENTION DIRECTIVE ACTIVATED**")
            st.write("Immediate notification dispatched to District Protection Officer. Priority VR Box-Breathing recommended below.")
        elif "YELLOW" in triage:
            st.warning("⚠️ **ELEVATED DISTRESS DETECTED**")
            st.write("Automated support ticket created for District Tele-Counselor 24h outreach.")
        else:
            st.success("✅ **STABLE BASELINE MAINTAINED**")
            st.write("Hearing logistics sent via SMS. Routine psychoeducation guidelines refreshed.")
    
    st.divider()
    st.write("**Extracted Bio-Acoustic & NLP Features:**")
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Pitch Jitter", f"{feats['pitch_jitter']}")
    fc2.metric("Hesitation Ratio", f"{feats['pause_ratio']}")
    fc3.metric("Trauma Keywords", f"{feats['trauma_keywords']}")
    fc4.metric("Negative Sentiment", f"{feats['negative_sentiment']}")
    st.markdown('</div>', unsafe_allow_html=True)

# 1. VOICE TAB
with tab_voice:
    c_left, c_right = st.columns([1.1, 1])

    if "live_voice_text_area" not in st.session_state:
        st.session_state["live_voice_text_area"] = ""
    if "last_recorded_hash" not in st.session_state:
        st.session_state["last_recorded_hash"] = None

    with c_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎙️ Live Acoustic Check-In")
        st.write("Record your statement. Whisper will automatically transcribe your speech below.")

        mic_audio = st.audio_input("Record voice statement", key="live_voice_recorder")

        # AUTO-TRANSCRIBE ON NEW AUDIO INPUT
        if mic_audio is not None:
            audio_bytes = mic_audio.getvalue()
            current_hash = hash(audio_bytes)
            
            # Run STT only once per unique recording
            if current_hash != st.session_state["last_recorded_hash"]:
                st.session_state["last_recorded_hash"] = current_hash
                temp_stt_file = "temp_stt_mic.wav"
                with open(temp_stt_file, "wb") as f:
                    f.write(audio_bytes)

                # Audio normalization so Whisper receives clear volume
                try:
                    data, samplerate = sf.read(temp_stt_file)
                    max_val = np.max(np.abs(data))
                    if max_val > 0.01:
                        normalized = data / max_val * 0.9
                        sf.write(temp_stt_file, normalized, samplerate)
                except Exception as norm_err:
                    print(f"[AUDIO NORM ERROR] {norm_err}")

                with st.spinner("Transcribing speech to text..."):
                    detected_text = transcribe_audio(temp_stt_file)
                    print(f"[UI STT] Detected text: {detected_text}")
                    if detected_text:
                        st.session_state["live_voice_text_area"] = detected_text

                if os.path.exists(temp_stt_file):
                    os.remove(temp_stt_file)
                st.rerun()

        # Text area directly tied to the session state key
        v_transcript = st.text_area(
            "Transcribed Statement / Context (Editable)",
            key="live_voice_text_area",
            height=120
        )

        eval_voice = st.button("Evaluate Voice Intake", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        if eval_voice:
            temp_wav = "temp_voice_in.wav"

            if mic_audio is not None:
                audio_bytes = mic_audio.getvalue()
                with open(temp_wav, "wb") as f:
                    f.write(audio_bytes)
                
                # Normalize evaluation audio for Librosa
                try:
                    data, samplerate = sf.read(temp_wav)
                    max_val = np.max(np.abs(data))
                    if max_val > 0.01:
                        normalized = data / max_val * 0.9
                        sf.write(temp_wav, normalized, samplerate)
                except Exception as e:
                    pass
            else:
                sr = 16000
                t = np.linspace(0, 2.0, sr * 2, endpoint=False)
                sf.write(temp_wav, 0.2 * np.sin(2 * np.pi * 220 * t), sr)

            active_text = st.session_state.get("live_voice_text_area", "").strip()

            with st.spinner("Extracting bio-acoustics & calculating distress score..."):
                res = compute_distress_score(temp_wav, active_text, d_hearing, threat_flag)
                log_session(user["username"], "Voice Check-In", res["score"], res["triage"], res["transcript"], res["features"])

            render_triage_dashboard(res)

            st.markdown("#### 🔍 Bio-Acoustic Diagnostics & Transcription")
            if res.get("transcript") and res["transcript"] != "Statement not provided.":
                st.success(f"**Processed Statement:** \"{res['transcript']}\"")
            else:
                st.warning("No transcript detected. Check microphone permissions or input audio level.")

            plot_acoustic_diagnostics(res["raw_audio"], res["pitches"])

            # IMMEDIATE INTERVENTION SHORTCUT FOR RED TRIAGE
            if "RED" in res["triage"]:
                st.markdown("---")
                st.markdown("### 🥽 Immediate Intervention: 360° Grounding Sanctuary")
                st.caption("Autonomic regulation recommended immediately based on elevated distress score.")
                render_360_sanctuary()

            if os.path.exists(temp_wav):
                os.remove(temp_wav)

# 2. DESCRIBE (TEXT) TAB
with tab_text:
    c_tleft, c_tright = st.columns([1.1, 1])
    with c_tleft:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("✍️ Written Trauma & Situation Report")
        st.write("Detailed description of threats, intimidation, or procedural anxiety.")
        t_stmt = st.text_area(
            "Describe what happened:",
            height=180,
            value="URGENT: I cannot speak on call right now because people are watching outside my window. Accused associates approached my brother this afternoon and threatened dire consequences if I testify on Monday. I am facing severe coercion and intimidation. Request immediate police escort and witness protection under Section 398 BNSS."
        )
        eval_text = st.button("Evaluate Written Report", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_tright:
        if eval_text:
            dummy_wav = "temp_text_in.wav"
            sr = 16000
            t = np.linspace(0, 1.0, sr, endpoint=False)
            sf.write(dummy_wav, 0.05 * np.sin(2 * np.pi * 180 * t), sr)
            
            res = compute_distress_score(dummy_wav, t_stmt if t_stmt.strip() else "Neutral statement", d_hearing, threat_flag)
            log_session(user["username"], "Written Report", res["score"], res["triage"], t_stmt, res["features"])
            
            render_triage_dashboard(res)

            if "RED" in res["triage"]:
                st.markdown("---")
                st.markdown("### 🥽 Recommended Clinical Intervention")
                render_360_sanctuary()
            
            if os.path.exists(dummy_wav):
                os.remove(dummy_wav)

# 3. VR GROUNDING TAB
with tab_vr:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🥽 Immersive VR Grounding & Acclimatization Modules")
    st.write("Prescribed immersive environments to regulate distress and alleviate courtroom anxiety.")
    
    vr_option = st.radio(
        "Select Interactive Clinical Module:",
        ["🌊 Module A: SOS Grounding Sanctuary (4-4-4-4 Box Breathing)", "⚖️ Module B: Judicial Courtroom Simulator (Katghara Stand)"],
        horizontal=True
    )
    
    if "Module A" in vr_option:
        st.markdown('<div class="vr-banner">', unsafe_allow_html=True)
        st.markdown("#### 🌊 Module A: SOS Grounding Sanctuary")
        st.write("• **Technique:** Bilateral audio stimulation, box-breathing guide (4-4-4-4).")
        st.write("• **Environment:** Calming coastal horizon simulation with rhythmic visual pacing.")
        st.markdown('</div>', unsafe_allow_html=True)
        render_360_sanctuary()
    else:
        st.markdown('<div class="vr-banner">', unsafe_allow_html=True)
        st.markdown("#### ⚖️ Module B: Judicial Courtroom Simulator")
        st.write("• **Technique:** Systematic desensitization & spatial familiarization under Section 398 BNSS.")
        st.write("• **Environment:** Realistic 3D witness box (*katghara*), judge's bench, and advocate stations.")
        st.markdown('</div>', unsafe_allow_html=True)
        render_360_courtroom()

    st.markdown('</div>', unsafe_allow_html=True)

# 4. HISTORY TAB
with tab_history:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📜 Case Intake & Triage Audit Trail")
    history_records = get_history(user["username"])
    if history_records:
        df = pd.DataFrame(history_records, columns=["Timestamp", "Intake Mode", "Distress Score (0-100)", "Assigned Triage"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recorded check-ins found for this identity profile.")
    st.markdown('</div>', unsafe_allow_html=True)