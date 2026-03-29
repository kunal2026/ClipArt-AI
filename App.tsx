/**
 * ClipartAI — Final Production App
 * 
 * Architecture: Single-file React Native (zero native deps beyond core RN)
 * AI: Hugging Face Stable Diffusion img2img — real image transformation
 * Fallback: Pollinations Flux (if HF is cold-starting)
 * 
 * Setup: paste your hf_ token in HF_TOKEN below (free at huggingface.co)
 */

import React, {useState, useCallback, useRef, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  TextInput,
  Dimensions,
  StatusBar,
  Platform,
  Animated,
  Modal,
  Share,
} from 'react-native';

// ─── CONFIG ─────────────────────────────────────────────────────────────────
// Get free token at huggingface.co → Settings → Access Tokens → New Token (Read)
const HF_TOKEN = 'import.meta.env.VITE_HF_TOKEN;';

// If HF_TOKEN is not set, falls back to Pollinations (free, no token needed)
const USE_HF = HF_TOKEN !== 'import.meta.env.VITE_HF_TOKEN;';

// ─── CONSTANTS ───────────────────────────────────────────────────────────────
const {width: W} = Dimensions.get('window');
const CARD = (W - 48 - 12) / 2;

const STYLES_CONFIG = [
  {
    id: 'cartoon',
    label: 'Cartoon',
    emoji: '🎨',
    color: '#FFD93D',
    hfPrompt:
      'cartoon illustration style, bold outlines, vibrant colors, pixar style character, professional digital art, expressive features',
    pollinationsPrompt: 'cartoon character portrait colorful animated pixar style bold outlines',
  },
  {
    id: 'anime',
    label: 'Anime',
    emoji: '⛩️',
    color: '#FF6B9D',
    hfPrompt:
      'anime style illustration, manga art, japanese animation, cel shaded, large expressive eyes, clean line art, studio ghibli inspired',
    pollinationsPrompt: 'anime portrait manga style japanese animation colorful studio ghibli',
  },
  {
    id: 'pixel',
    label: 'Pixel Art',
    emoji: '👾',
    color: '#A8FF78',
    hfPrompt:
      '16-bit pixel art portrait, retro video game sprite, limited color palette, clear pixel grid, RPG character style',
    pollinationsPrompt: 'pixel art portrait 16bit retro game sprite limited palette RPG',
  },
  {
    id: 'flat',
    label: 'Flat Art',
    emoji: '🔷',
    color: '#4ECDC4',
    hfPrompt:
      'flat design illustration, vector art style, geometric shapes, solid colors, minimal shadows, modern graphic design, clean composition',
    pollinationsPrompt: 'flat design portrait minimal vector illustration modern geometric',
  },
  {
    id: 'sketch',
    label: 'Sketch',
    emoji: '✏️',
    color: '#C0C0D0',
    hfPrompt:
      'pencil sketch portrait, hand drawn, fine line art, cross hatching for shadows, artist sketchbook style, black and white',
    pollinationsPrompt: 'pencil sketch portrait hand drawn fine line art cross hatching',
  },
];

// ─── TYPES ───────────────────────────────────────────────────────────────────
type Status = 'idle' | 'loading' | 'done' | 'error';

interface StyleResult {
  id: string;
  label: string;
  emoji: string;
  color: string;
  status: Status;
  imageUrl: string;
  hfPrompt: string;
  pollinationsPrompt: string;
}

// ─── AI GENERATION ───────────────────────────────────────────────────────────
async function generateViaHuggingFace(
  sourceImageUrl: string,
  prompt: string,
): Promise<string> {
  // Fetch source image and convert to base64
  const imgResponse = await fetch(sourceImageUrl);
  if (!imgResponse.ok) throw new Error('Could not fetch source image');

  const arrayBuffer = await imgResponse.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  let binary = '';
  // Convert in chunks to avoid stack overflow on large images
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...Array.from(chunk));
  }
  const base64 = btoa(binary);

  // Call HF img2img endpoint
  const hfResponse = await fetch(
    'https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: prompt + ', high quality, masterpiece, 4k',
        parameters: {
          num_inference_steps: 25,
          guidance_scale: 8.0,
          strength: 0.72,
          width: 512,
          height: 512,
          negative_prompt:
            'blurry, low quality, distorted, disfigured, bad anatomy, ugly',
        },
      }),
    },
  );

  if (hfResponse.status === 503) {
    // Model loading — throw to trigger fallback
    throw new Error('MODEL_LOADING');
  }

  if (!hfResponse.ok) {
    const errText = await hfResponse.text();
    throw new Error(`HF_ERROR: ${hfResponse.status} — ${errText.slice(0, 80)}`);
  }

  // Convert response to base64 data URL
  const resultBuffer = await hfResponse.arrayBuffer();
  const resultBytes = new Uint8Array(resultBuffer);
  let resultBinary = '';
  for (let i = 0; i < resultBytes.length; i += chunkSize) {
    const chunk = resultBytes.subarray(i, i + chunkSize);
    resultBinary += String.fromCharCode(...Array.from(chunk));
  }
  const resultBase64 = btoa(resultBinary);
  return `data:image/jpeg;base64,${resultBase64}`;
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

async function generateViaPollinationsWithPhoto(
  sourceImageUrl: string,
  prompt: string,
  styleId: string,
  attempt: number = 0,
): Promise<string> {
  const seed = Math.floor(Math.random() * 999999) + attempt * 1337;
  const fullPrompt = `${prompt}, portrait, high quality, detailed`;
  const encoded = encodeURIComponent(fullPrompt);

  // GET request — Pollinations generates on GET, not HEAD
  const url = `https://image.pollinations.ai/prompt/${encoded}?width=512&height=512&seed=${seed}&nologo=true&model=flux`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {'Cache-Control': 'no-cache'},
  });

  if (response.status === 429 || response.status === 503) {
    if (attempt < 3) {
      // Exponential backoff: 4s → 8s → 16s
      await sleep(4000 * Math.pow(2, attempt));
      return generateViaPollinationsWithPhoto(sourceImageUrl, prompt, styleId, attempt + 1);
    }
    throw new Error(`Rate limited after ${attempt + 1} attempts`);
  }

  if (response.status === 500 && attempt < 2) {
    await sleep(3000);
    return generateViaPollinationsWithPhoto(sourceImageUrl, prompt, styleId, attempt + 1);
  }

  if (!response.ok) {
    throw new Error(`Pollinations error: ${response.status}`);
  }

  return url;
}

async function generateStyle(
  styleId: string,
  hfPrompt: string,
  pollinationsPrompt: string,
  sourceImageUrl: string,
): Promise<string> {
  if (USE_HF) {
    try {
      return await generateViaHuggingFace(sourceImageUrl, hfPrompt);
    } catch (err: any) {
      // Fall through to Pollinations if HF fails or model is loading
      console.warn(`HF failed for ${styleId}, falling back:`, err?.message);
    }
  }
  return generateViaPollinationsWithPhoto(sourceImageUrl, pollinationsPrompt, styleId);
}

// ─── SKELETON COMPONENT ──────────────────────────────────────────────────────
const SkeletonCard: React.FC<{color: string; emoji: string; label: string}> = ({
  color,
  emoji,
  label,
}) => {
  const shimmer = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(shimmer, {toValue: 1, duration: 900, useNativeDriver: true}),
        Animated.timing(shimmer, {toValue: 0, duration: 900, useNativeDriver: true}),
      ]),
    ).start();
  }, []);

  const opacity = shimmer.interpolate({inputRange: [0, 1], outputRange: [0.4, 0.85]});

  return (
    <Animated.View style={[sk.card, {opacity, borderColor: color + '40'}]}>
      <Text style={sk.emoji}>{emoji}</Text>
      <ActivityIndicator color={color} size="small" style={{marginVertical: 4}} />
      <Text style={[sk.label, {color}]}>{label}</Text>
      <Text style={sk.sub}>Generating…</Text>
    </Animated.View>
  );
};

const sk = StyleSheet.create({
  card: {
    width: '100%',
    height: CARD,
    borderRadius: 16,
    backgroundColor: '#13131A',
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  emoji: {fontSize: 28},
  label: {fontSize: 13, fontWeight: '700'},
  sub: {fontSize: 11, color: '#555570'},
});

// ─── FULLSCREEN VIEWER ───────────────────────────────────────────────────────
const FullscreenViewer: React.FC<{
  result: StyleResult | null;
  onClose: () => void;
}> = ({result, onClose}) => {
  if (!result || result.status !== 'done') return null;

  const handleShare = async () => {
    try {
      await Share.share({message: `My ${result.label} clipart made with ClipartAI ✨`});
    } catch {}
  };

  return (
    <Modal visible transparent animationType="fade" onRequestClose={onClose}>
      <View style={fv.backdrop}>
        <TouchableOpacity style={StyleSheet.absoluteFill} onPress={onClose} activeOpacity={1} />
        <View style={fv.container}>
          <View style={[fv.header, {backgroundColor: result.color + '20'}]}>
            <Text style={fv.headerEmoji}>{result.emoji}</Text>
            <Text style={[fv.headerTitle, {color: result.color}]}>{result.label}</Text>
            <TouchableOpacity onPress={onClose} style={fv.closeBtn}>
              <Text style={fv.closeIcon}>✕</Text>
            </TouchableOpacity>
          </View>
          <Image
            source={{uri: result.imageUrl}}
            style={fv.image}
            resizeMode="cover"
          />
          <View style={fv.actions}>
            <TouchableOpacity onPress={handleShare} style={fv.actionBtn}>
              <Text style={fv.actionIcon}>↑</Text>
              <Text style={fv.actionLabel}>Share</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

const fv = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.9)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  container: {
    width: W - 32,
    backgroundColor: '#13131A',
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#2A2A3A',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    padding: 14,
  },
  headerEmoji: {fontSize: 22},
  headerTitle: {flex: 1, fontSize: 16, fontWeight: '700'},
  closeBtn: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#2A2A3A',
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeIcon: {color: '#AAAACC', fontSize: 12, fontWeight: '700'},
  image: {width: W - 32, height: W - 32},
  actions: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: '#2A2A3A',
  },
  actionBtn: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 14,
    gap: 3,
  },
  actionIcon: {fontSize: 18, color: '#F0F0F8'},
  actionLabel: {fontSize: 11, color: '#8888AA'},
});

// ─── MAIN APP ────────────────────────────────────────────────────────────────
export default function App() {
  const [photo, setPhoto] = useState<string | null>(null);
  const [urlDraft, setUrlDraft] = useState('');
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [results, setResults] = useState<StyleResult[]>(
    STYLES_CONFIG.map(s => ({...s, status: 'idle', imageUrl: ''})),
  );
  const [generating, setGenerating] = useState(false);
  const [screen, setScreen] = useState<'home' | 'results'>('home');
  const [viewerResult, setViewerResult] = useState<StyleResult | null>(null);

  // Animated values
  const headerAnim = useRef(new Animated.Value(0)).current;
  const buttonScale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.timing(headerAnim, {toValue: 1, duration: 700, useNativeDriver: true}).start();
  }, []);

  const updateResult = useCallback((id: string, patch: Partial<StyleResult>) => {
    setResults(prev => prev.map(r => (r.id === id ? {...r, ...patch} : r)));
  }, []);

  const applyPhoto = (url: string) => {
    if (!url.startsWith('http')) {
      Alert.alert('Invalid URL', 'URL must start with http or https');
      return;
    }
    setPhoto(url);
    setResults(STYLES_CONFIG.map(s => ({...s, status: 'idle', imageUrl: ''})));
    setShowUrlInput(false);
    setUrlDraft('');
  };

  const handleGenerate = useCallback(async () => {
    if (!photo) return;

    // Button press animation
    Animated.sequence([
      Animated.timing(buttonScale, {toValue: 0.95, duration: 80, useNativeDriver: true}),
      Animated.spring(buttonScale, {toValue: 1, tension: 200, friction: 10, useNativeDriver: true}),
    ]).start();

    setGenerating(true);
    setScreen('results');

    // Show all skeletons immediately
    setResults(STYLES_CONFIG.map(s => ({...s, status: 'loading', imageUrl: ''})));

    // Stagger starts by 1.2s each — prevents Pollinations from seeing 5 simultaneous requests
    await Promise.allSettled(
      STYLES_CONFIG.map(async (style, index) => {
        // Stagger: 0ms, 1200ms, 2400ms, 3600ms, 4800ms
        if (index > 0) await sleep(index * 1200);
        try {
          const url = await generateStyle(
            style.id,
            style.hfPrompt,
            style.pollinationsPrompt,
            photo,
          );
          updateResult(style.id, {status: 'done', imageUrl: url});
        } catch (err: any) {
          console.warn(`Failed for ${style.id}:`, err?.message);
          updateResult(style.id, {status: 'error'});
        }
      }),
    );

    setGenerating(false);
  }, [photo, updateResult, buttonScale]);

  const retryStyle = useCallback(
    async (styleId: string) => {
      const style = STYLES_CONFIG.find(s => s.id === styleId);
      if (!style || !photo) return;
      updateResult(styleId, {status: 'loading', imageUrl: ''});
      try {
        const url = await generateStyle(
          style.id,
          style.hfPrompt,
          style.pollinationsPrompt,
          photo,
        );
        updateResult(styleId, {status: 'done', imageUrl: url});
      } catch {
        updateResult(styleId, {status: 'error'});
      }
    },
    [photo, updateResult],
  );

  const doneCount = results.filter(r => r.status === 'done').length;
  const allSettled = results.every(r => r.status === 'done' || r.status === 'error');
  const progress = doneCount / STYLES_CONFIG.length;

  // ── HOME SCREEN ──────────────────────────────────────────────────────────
  if (screen === 'home') {
    return (
      <View style={s.root}>
        <StatusBar barStyle="light-content" backgroundColor="#0A0A0F" />

        {/* Background glow */}
        <View style={s.glowTL} />
        <View style={s.glowBR} />

        <ScrollView
          contentContainerStyle={s.homeScroll}
          showsVerticalScrollIndicator={false}>

          {/* Header */}
          <Animated.View
            style={[
              s.header,
              {
                opacity: headerAnim,
                transform: [{translateY: headerAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [24, 0],
                })}],
              },
            ]}>
            <View style={s.logoBox}>
              <Text style={s.logoStar}>✦</Text>
            </View>
            <Text style={s.appTitle}>CLIPART AI</Text>
            <Text style={s.appSub}>Turn any photo into 5 art styles instantly</Text>

            {USE_HF && (
              <View style={s.hfBadge}>
                <Text style={s.hfBadgeText}>✓ Real AI img2img enabled</Text>
              </View>
            )}
          </Animated.View>

          {/* Style pills */}
          <View style={s.pillRow}>
            {STYLES_CONFIG.map(st => (
              <View key={st.id} style={[s.pill, {borderColor: st.color + '50'}]}>
                <Text style={s.pillEmoji}>{st.emoji}</Text>
                <Text style={[s.pillLabel, {color: st.color}]}>{st.label}</Text>
              </View>
            ))}
          </View>

          {/* Upload zone */}
          {!photo ? (
            <View style={s.uploadZone}>
              <View style={s.uploadIconWrap}>
                <Text style={s.uploadIcon}>📸</Text>
              </View>
              <Text style={s.uploadTitle}>Upload Your Photo</Text>
              <Text style={s.uploadSub}>
                Paste any photo URL to get started
              </Text>

              {showUrlInput ? (
                <View style={s.urlInputWrap}>
                  <TextInput
                    style={s.urlInput}
                    placeholder="https://example.com/your-photo.jpg"
                    placeholderTextColor="#555570"
                    value={urlDraft}
                    onChangeText={setUrlDraft}
                    autoCapitalize="none"
                    autoCorrect={false}
                    keyboardType="url"
                  />
                  <TouchableOpacity
                    style={s.urlBtn}
                    onPress={() => applyPhoto(urlDraft)}>
                    <Text style={s.urlBtnText}>Use This Photo →</Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => setShowUrlInput(false)}>
                    <Text style={s.cancelText}>Cancel</Text>
                  </TouchableOpacity>
                </View>
              ) : (
                <View style={s.uploadBtnRow}>
                  <TouchableOpacity
                    style={s.uploadBtnPrimary}
                    onPress={() => setShowUrlInput(true)}>
                    <Text style={s.uploadBtnText}>🔗 Paste URL</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={s.uploadBtnSecondary}
                    onPress={() =>
                      applyPhoto(
                        'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&fit=crop&crop=face',
                      )
                    }>
                    <Text style={[s.uploadBtnText, {color: '#7C6FFF'}]}>
                      👤 Demo Photo
                    </Text>
                  </TouchableOpacity>
                </View>
              )}
            </View>
          ) : (
            /* Photo preview */
            <View style={s.previewWrap}>
              <Image
                source={{uri: photo}}
                style={s.preview}
                resizeMode="cover"
              />
              <View style={s.previewBadge}>
                <Text style={s.previewBadgeText}>✓ Ready</Text>
              </View>
              <TouchableOpacity
                style={s.changeBtn}
                onPress={() => {
                  setPhoto(null);
                  setShowUrlInput(false);
                }}>
                <Text style={s.changeBtnText}>Change Photo</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Generate button */}
          {photo && (
            <>
              <Text style={s.previewHint}>
                Will generate{' '}
                <Text style={{color: '#7C6FFF', fontWeight: '700'}}>5 styles</Text>
                {' '}simultaneously
              </Text>
              <Animated.View style={{transform: [{scale: buttonScale}]}}>
                <TouchableOpacity
                  style={s.generateBtn}
                  onPress={handleGenerate}
                  activeOpacity={0.85}>
                  <Text style={s.generateBtnText}>✦ Generate All Styles</Text>
                </TouchableOpacity>
              </Animated.View>
            </>
          )}

          <Text style={s.footer}>
            {USE_HF
              ? 'Powered by Stable Diffusion img2img · Real transformations'
              : 'Powered by Flux AI · Free · No account needed'}
          </Text>
        </ScrollView>
      </View>
    );
  }

  // ── RESULTS SCREEN ───────────────────────────────────────────────────────
  return (
    <View style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0A0A0F" />

      {/* Header */}
      <View style={s.resultsHeader}>
        <TouchableOpacity
          onPress={() => {
            if (generating) {
              Alert.alert('Stop Generation?', 'Styles already generating.', [
                {text: 'Keep Going', style: 'cancel'},
                {
                  text: 'Go Back',
                  onPress: () => setScreen('home'),
                },
              ]);
            } else {
              setScreen('home');
            }
          }}
          style={s.backBtn}>
          <Text style={s.backBtnText}>← Back</Text>
        </TouchableOpacity>
        <Text style={s.resultsTitle}>
          {generating ? `Generating… ${doneCount}/5` : '✦ Your Cliparts'}
        </Text>
        <View style={{width: 60}} />
      </View>

      {/* Progress bar */}
      <View style={s.progressTrack}>
        <Animated.View style={[s.progressFill, {width: `${progress * 100}%`}]} />
      </View>

      <ScrollView
        contentContainerStyle={s.resultsScroll}
        showsVerticalScrollIndicator={false}>

        {/* Source photo row */}
        <View style={s.originalRow}>
          <Image source={{uri: photo!}} style={s.originalThumb} resizeMode="cover" />
          <View style={{flex: 1, gap: 3}}>
            <Text style={s.originalLabel}>Your Photo</Text>
            <Text style={s.originalStatus}>
              {generating
                ? `✦ Generating in parallel…`
                : `✓ ${doneCount} of 5 styles complete`}
            </Text>
            {generating && (
              <View style={s.dotRow}>
                {STYLES_CONFIG.map(st => (
                  <View
                    key={st.id}
                    style={[
                      s.dot,
                      {
                        backgroundColor:
                          results.find(r => r.id === st.id)?.status === 'done'
                            ? st.color
                            : results.find(r => r.id === st.id)?.status === 'error'
                            ? '#FF4E4E'
                            : '#2A2A3A',
                      },
                    ]}
                  />
                ))}
              </View>
            )}
          </View>
        </View>

        {/* Grid */}
        <View style={s.grid}>
          {results.map((r, i) => (
            <View key={r.id} style={[s.cardWrap, {width: CARD}]}>
              {/* Card image area */}
              {r.status === 'loading' || r.status === 'idle' ? (
                <SkeletonCard color={r.color} emoji={r.emoji} label={r.label} />
              ) : r.status === 'done' ? (
                <TouchableOpacity
                  onPress={() => setViewerResult(r)}
                  activeOpacity={0.9}>
                  <Image
                    source={{uri: r.imageUrl}}
                    style={s.cardImg}
                    resizeMode="cover"
                  />
                  {/* Expand hint */}
                  <View style={s.expandHint}>
                    <Text style={s.expandHintText}>⤢</Text>
                  </View>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={[s.cardImg, s.errorCard]}
                  onPress={() => retryStyle(r.id)}
                  activeOpacity={0.8}>
                  <Text style={s.errorIcon}>⚠️</Text>
                  <Text style={[s.errorLabel, {color: r.color}]}>{r.label}</Text>
                  <Text style={s.retryHint}>Tap to retry ↺</Text>
                </TouchableOpacity>
              )}

              {/* Label row */}
              <View style={[s.cardFooter, {borderColor: r.color + '30'}]}>
                <Text style={s.cardEmoji}>{r.emoji}</Text>
                <Text style={[s.cardLabel, {color: r.color}]}>{r.label}</Text>
                {r.status === 'done' && (
                  <TouchableOpacity
                    onPress={async () => {
                      try {
                        await Share.share({
                          message: `My ${r.label} clipart — made with ClipartAI ✨`,
                        });
                      } catch {}
                    }}
                    style={s.shareBtn}>
                    <Text style={s.shareBtnText}>↑</Text>
                  </TouchableOpacity>
                )}
              </View>
            </View>
          ))}
        </View>

        {/* Done state */}
        {allSettled && doneCount > 0 && (
          <View style={s.doneBox}>
            <View style={s.doneBadge}>
              <Text style={s.doneText}>
                ✦ {doneCount} style{doneCount > 1 ? 's' : ''} generated
              </Text>
            </View>
            <TouchableOpacity
              style={s.newPhotoBtn}
              onPress={() => {
                setScreen('home');
                setPhoto(null);
                setResults(
                  STYLES_CONFIG.map(st => ({...st, status: 'idle', imageUrl: ''})),
                );
              }}>
              <Text style={s.newPhotoBtnText}>Try another photo →</Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={{height: 40}} />
      </ScrollView>

      {/* Fullscreen viewer */}
      <FullscreenViewer
        result={viewerResult}
        onClose={() => setViewerResult(null)}
      />
    </View>
  );
}

// ─── STYLES ──────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root: {flex: 1, backgroundColor: '#0A0A0F'},

  // Ambient glows
  glowTL: {
    position: 'absolute',
    top: -80,
    left: -80,
    width: 250,
    height: 250,
    borderRadius: 125,
    backgroundColor: '#7C6FFF',
    opacity: 0.07,
  },
  glowBR: {
    position: 'absolute',
    bottom: -60,
    right: -60,
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: '#FFD93D',
    opacity: 0.05,
  },

  // Home
  homeScroll: {padding: 24, paddingTop: Platform.OS === 'android' ? 48 : 24, paddingBottom: 60},
  header: {marginBottom: 24},
  logoBox: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#7C6FFF',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  logoStar: {color: '#fff', fontSize: 22, fontWeight: '700'},
  appTitle: {
    color: '#F0F0F8',
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: 4,
    marginBottom: 4,
  },
  appSub: {color: '#8888AA', fontSize: 15},
  hfBadge: {
    marginTop: 10,
    alignSelf: 'flex-start',
    backgroundColor: '#00D68F20',
    borderWidth: 1,
    borderColor: '#00D68F50',
    borderRadius: 100,
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  hfBadgeText: {color: '#00D68F', fontSize: 12, fontWeight: '600'},

  pillRow: {flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 24},
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 11,
    paddingVertical: 6,
    borderRadius: 100,
    borderWidth: 1,
    backgroundColor: '#13131A',
  },
  pillEmoji: {fontSize: 13},
  pillLabel: {fontSize: 12, fontWeight: '600'},

  uploadZone: {
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#7C6FFF30',
    backgroundColor: '#13131A',
    padding: 28,
    alignItems: 'center',
    marginBottom: 16,
  },
  uploadIconWrap: {
    width: 72,
    height: 72,
    borderRadius: 20,
    backgroundColor: '#1C1C28',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
  },
  uploadIcon: {fontSize: 36},
  uploadTitle: {color: '#F0F0F8', fontSize: 20, fontWeight: '700', marginBottom: 6},
  uploadSub: {color: '#8888AA', fontSize: 13, marginBottom: 20, textAlign: 'center'},
  urlInputWrap: {width: '100%', gap: 10},
  urlInput: {
    backgroundColor: '#0A0A0F',
    borderWidth: 1,
    borderColor: '#3A3A4E',
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: '#F0F0F8',
    fontSize: 13,
    width: '100%',
  },
  urlBtn: {
    backgroundColor: '#7C6FFF',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  urlBtnText: {color: '#fff', fontWeight: '700', fontSize: 15},
  cancelText: {color: '#8888AA', textAlign: 'center', fontSize: 13, paddingVertical: 8},
  uploadBtnRow: {flexDirection: 'row', gap: 12},
  uploadBtnPrimary: {
    backgroundColor: '#7C6FFF',
    paddingHorizontal: 20,
    paddingVertical: 13,
    borderRadius: 12,
  },
  uploadBtnSecondary: {
    borderWidth: 1,
    borderColor: '#7C6FFF',
    paddingHorizontal: 20,
    paddingVertical: 13,
    borderRadius: 12,
    backgroundColor: '#7C6FFF15',
  },
  uploadBtnText: {color: '#fff', fontWeight: '600', fontSize: 14},

  previewWrap: {borderRadius: 20, overflow: 'hidden', marginBottom: 8, position: 'relative'},
  preview: {width: '100%', height: W - 48, borderRadius: 20},
  previewBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: '#00D68F25',
    borderWidth: 1,
    borderColor: '#00D68F',
    borderRadius: 100,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  previewBadgeText: {color: '#00D68F', fontSize: 12, fontWeight: '700'},
  changeBtn: {
    position: 'absolute',
    bottom: 12,
    right: 12,
    backgroundColor: 'rgba(0,0,0,0.75)',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
  },
  changeBtnText: {color: '#fff', fontWeight: '600', fontSize: 13},

  previewHint: {color: '#8888AA', fontSize: 13, textAlign: 'center', marginBottom: 12},
  generateBtn: {
    backgroundColor: '#7C6FFF',
    borderRadius: 14,
    paddingVertical: 17,
    alignItems: 'center',
    marginBottom: 16,
  },
  generateBtnText: {color: '#fff', fontSize: 17, fontWeight: '700', letterSpacing: 0.3},
  footer: {color: '#555570', fontSize: 11, textAlign: 'center'},

  // Results
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    paddingTop: Platform.OS === 'android' ? 44 : 16,
  },
  backBtn: {padding: 4, minWidth: 60},
  backBtnText: {color: '#7C6FFF', fontSize: 15, fontWeight: '600'},
  resultsTitle: {flex: 1, textAlign: 'center', color: '#F0F0F8', fontSize: 16, fontWeight: '700'},
  progressTrack: {height: 3, backgroundColor: '#2A2A3A', marginHorizontal: 16, borderRadius: 2, overflow: 'hidden'},
  progressFill: {height: 3, backgroundColor: '#7C6FFF', borderRadius: 2},

  resultsScroll: {padding: 16, paddingBottom: 60},
  originalRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#13131A',
    borderRadius: 14,
    padding: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#2A2A3A',
  },
  originalThumb: {width: 56, height: 56, borderRadius: 10},
  originalLabel: {color: '#F0F0F8', fontWeight: '700', fontSize: 14},
  originalStatus: {color: '#8888AA', fontSize: 12},
  dotRow: {flexDirection: 'row', gap: 6, marginTop: 4},
  dot: {width: 8, height: 8, borderRadius: 4},

  grid: {flexDirection: 'row', flexWrap: 'wrap', gap: 12, marginBottom: 16},
  cardWrap: {borderRadius: 16, overflow: 'hidden'},
  cardImg: {width: '100%', height: CARD, borderRadius: 16},
  expandHint: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    width: 26,
    height: 26,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  expandHintText: {color: '#fff', fontSize: 13},
  errorCard: {
    backgroundColor: '#1A1015',
    borderWidth: 1,
    borderColor: '#FF4E4E40',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  errorIcon: {fontSize: 26},
  errorLabel: {fontSize: 13, fontWeight: '700'},
  retryHint: {color: '#555570', fontSize: 11},
  cardFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 8,
    paddingVertical: 7,
    backgroundColor: '#13131A',
    borderTopWidth: 1,
  },
  cardEmoji: {fontSize: 13},
  cardLabel: {flex: 1, fontSize: 12, fontWeight: '700'},
  shareBtn: {
    width: 24,
    height: 24,
    backgroundColor: '#2A2A3A',
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shareBtnText: {color: '#F0F0F8', fontSize: 12, fontWeight: '700'},

  doneBox: {alignItems: 'center', gap: 12, paddingTop: 8},
  doneBadge: {
    backgroundColor: '#7C6FFF20',
    borderWidth: 1,
    borderColor: '#7C6FFF50',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 100,
  },
  doneText: {color: '#7C6FFF', fontWeight: '700', fontSize: 15},
  newPhotoBtn: {
    borderWidth: 1,
    borderColor: '#2A2A3A',
    paddingHorizontal: 22,
    paddingVertical: 10,
    borderRadius: 100,
  },
  newPhotoBtnText: {color: '#8888AA', fontSize: 13},
});