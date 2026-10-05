import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import {
  Alert,
  BackHandler,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  Switch,
  Text,
  View,
  useColorScheme,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { router, Slot, usePathname } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Picker from 'expo-image-picker';
import * as Manipulator from 'expo-image-manipulator';
import {
  Quicksand_400Regular,
  Quicksand_500Medium,
  Quicksand_600SemiBold,
  Quicksand_700Bold,
  useFonts,
} from '@expo-google-fonts/quicksand';
import Motion from './Motion';
import {
  AnalyzeArt,
  AvatarArt,
  Brand,
  Button,
  Card,
  Field,
  FoodPhoneArt,
  Icon,
  LogoMark,
  MacroDonut,
  ServerBadge,
  SplashMealArt,
  T,
  Theme,
  dark,
  light,
  type Palette,
} from './components';
import {
  api,
  analyze,
  baseURL,
  onExpired,
  restoreToken,
  setURL,
  storeToken,
  type Profile,
  type Result,
  type User,
} from './api';

type Screen =
  | 'welcome'
  | 'intro'
  | 'login'
  | 'signup'
  | 'onboarding'
  | 'home'
  | 'scan'
  | 'analyzing'
  | 'detected'
  | 'result'
  | 'insights'
  | 'history'
  | 'profile'
  | 'settings';

type Dashboard = { total_scans: number; recent: Result[] };

type DraftProfile = {
  name: string;
  age: string;
  sex: string;
  height: string;
  weight: string;
  allergies: string;
  conditions: string;
  preferences: string[];
  avoid: string;
  goal: string;
};

const AppContext = createContext<{ c: Palette }>({ c: light });

const nice = (value: string) =>
  value
    .replace(/^(BD_|staple-|meatonly-)/, '')
    .replace(/chickenleg/gi, 'chicken leg')
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());

const splitCsv = (value: string) =>
  value
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

const emptyDraft = (name = ''): DraftProfile => ({
  name,
  age: '',
  sex: '',
  height: '',
  weight: '',
  allergies: '',
  conditions: '',
  preferences: [],
  avoid: '',
  goal: 'Build mindful habits',
});

function profileComplete(user: User | null) {
  if (!user) return false;
  const profile = user.profile || {};
  return profile.onboarding_complete === true || Object.keys(profile).length > 2;
}

export default function App() {
  return (
    <SafeAreaProvider>
      <Slot />
      <Main />
    </SafeAreaProvider>
  );
}

function Main() {
  const [fonts, fontError] = useFonts({
    Quicksand_400Regular,
    Quicksand_500Medium,
    Quicksand_600SemiBold,
    Quicksand_700Bold,
  });
  const system = useColorScheme();
  const pathname = usePathname();
  const { width } = useWindowDimensions();
  const [mode, setMode] = useState<'system' | 'light' | 'dark'>('system');
  const c = mode === 'dark' || (mode === 'system' && system === 'dark') ? dark : light;
  const screen = (pathname === '/' ? 'welcome' : pathname.slice(1)) as Screen;
  const scrollRef = useRef<ScrollView | null>(null);
  const busyLock = useRef(false);

  const [ready, setReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [draft, setDraft] = useState<DraftProfile>(emptyDraft());
  const [uri, setUri] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [history, setHistory] = useState<Result[]>([]);
  const [moreHistory, setMoreHistory] = useState(false);
  const [dashboard, setDashboard] = useState<Dashboard>({ total_scans: 0, recent: [] });
  const [showBoxes, setShowBoxes] = useState(true);
  const [serverURL, setServerURL] = useState(baseURL);
  const [serverState, setServerState] = useState<'checking' | 'online' | 'offline'>('checking');
  const [health, setHealth] = useState<any>(null);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');

  const contentWidth = Math.min(width, 470);
  const compact = width < 380;

  const setScreen = (next: Screen) => {
    router.replace((next === 'welcome' ? '/' : `/${next}`) as any);
  };

  const go = (next: Screen) => {
    setError('');
    setMessage('');
    setScreen(next);
  };

  useEffect(() => {
    scrollRef.current?.scrollTo({ y: 0, animated: false });
  }, [screen]);

  useEffect(
    () =>
      onExpired(() => {
        setUser(null);
        setHistory([]);
        setResult(null);
        setUri('');
        setMessage('Your session expired. Please sign in again.');
        setScreen('login');
      }),
    [],
  );

  useEffect(() => {
    (async () => {
      try {
        const saved = await AsyncStorage.multiGet(['nutrisight-theme', 'nutrisight-server']);
        const savedMode = saved[0][1] as 'system' | 'light' | 'dark' | null;
        if (savedMode && ['system', 'light', 'dark'].includes(savedMode)) setMode(savedMode);
        if (saved[1][1]) {
          setURL(saved[1][1]);
          setServerURL(saved[1][1]);
        }
        if (await restoreToken()) {
          try {
            const restored = await api<User>('/api/me');
            acceptUser(restored);
            setScreen(profileComplete(restored) ? 'home' : 'onboarding');
          } catch {
            setScreen('login');
          }
        }
      } finally {
        setReady(true);
      }
    })();
  }, []);

  useEffect(() => {
    if (!ready) return;
    checkServer(false);
  }, [ready, serverURL]);

  useEffect(() => {
    if (screen === 'home' && user) {
      api<Dashboard>('/api/dashboard').then(setDashboard).catch(() => {});
    }
  }, [screen, user?.id]);

  useEffect(() => {
    let active = true;
    if (screen === 'history' && user) {
      api<Result[]>('/api/history')
        .then((rows) => {
          if (!active) return;
          setHistory(rows);
          setMoreHistory(rows.length === 50);
        })
        .catch((e) => active && setError(e.message));
    }
    return () => {
      active = false;
    };
  }, [screen, user?.id]);

  useEffect(() => {
    if (Platform.OS !== 'android') return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      if (busy || screen === 'analyzing') return true;
      if (['login', 'signup'].includes(screen)) {
        go('intro');
        return true;
      }
      if (screen === 'onboarding') {
        return true;
      }
      if (screen === 'detected') { go('scan'); return true; }
      if (screen === 'result') { go('detected'); return true; }
      if (screen === 'insights') { go('result'); return true; }
      if (!['home', 'welcome', 'intro'].includes(screen)) {
        go('home');
        return true;
      }
      return false;
    });
    return () => sub.remove();
  }, [screen, busy]);

  function acceptUser(next: User) {
    setUser(next);
    const p = next.profile || {};
    setDraft({
      name: next.name || '',
      age: p.age ? String(p.age) : '',
      sex: p.sex || '',
      height: p.height_cm ? String(p.height_cm) : '',
      weight: p.weight_kg ? String(p.weight_kg) : '',
      allergies: (p.allergies || []).join(', '),
      conditions: (p.conditions || []).join(', '),
      preferences: p.preferences || [],
      avoid: (p.avoid || []).join(', '),
      goal: p.goal || 'Build mindful habits',
    });
  }

  async function run(task: () => Promise<void>) {
    if (busyLock.current) return;
    busyLock.current = true;
    setBusy(true);
    setError('');
    setMessage('');
    try {
      await task();
    } catch (e: any) {
      setError(e?.message || 'Something went wrong. Please try again.');
    } finally {
      setBusy(false);
      busyLock.current = false;
    }
  }

  async function auth(kind: 'login' | 'signup') {
    await run(async () => {
      if (kind === 'signup' && !name.trim()) throw new Error('Please enter your name.');
      if (!email.trim()) throw new Error('Please enter your email address.');
      if (password.length < 8) throw new Error('Use at least 8 characters for your password.');
            const data = await api<{ token: string; user: User }>(`/api/auth/${kind}`, 'POST', {
        email,
        password,
        name,
      });
      await storeToken(data.token);
      acceptUser(data.user);
      setPassword('');
      setConfirmPassword('');
      setResult(null);
      setUri('');
      setHistory([]);
      setDashboard({ total_scans: 0, recent: [] });
      setScreen(profileComplete(data.user) ? 'home' : 'onboarding');
    });
  }

  async function saveProfile(destination: Screen = 'home') {
    await run(async () => {
      const age = draft.age ? Number(draft.age) : null;
      const height = draft.height ? Number(draft.height) : null;
      const weight = draft.weight ? Number(draft.weight) : null;
      if (age !== null && (!Number.isFinite(age) || age < 1 || age > 120)) throw new Error('Enter an age between 1 and 120.');
      if (height !== null && (!Number.isFinite(height) || height < 50 || height > 260)) throw new Error('Enter a height between 50 and 260 cm.');
      if (weight !== null && (!Number.isFinite(weight) || weight < 15 || weight > 400)) throw new Error('Enter a weight between 15 and 400 kg.');
      const payload: Profile = {
        name: draft.name.trim(),
        age,
        sex: draft.sex || null,
        height_cm: height,
        weight_kg: weight,
        allergies: splitCsv(draft.allergies),
        conditions: splitCsv(draft.conditions),
        preferences: draft.preferences,
        avoid: splitCsv(draft.avoid),
        goal: draft.goal,
      };
      const updated = await api<User>('/api/profile', 'PUT', payload);
      acceptUser(updated);
      setMessage('Your nutrition profile is saved.');
      setScreen(destination);
    });
  }

  async function skipOnboarding() {
    await run(async () => {
      const updated = await api<User>('/api/profile/skip', 'POST', {});
      acceptUser(updated);
      setScreen('home');
    });
  }

  async function pick(camera: boolean) {
    await run(async () => {
      if (camera) {
        const permission = await Picker.requestCameraPermissionsAsync();
        if (!permission.granted) throw new Error('Camera permission is required to take a meal photo.');
      }
      const options: Picker.ImagePickerOptions = {
        mediaTypes: ['images'],
        quality: 0.9,
        allowsEditing: false,
      };
      const photo = camera ? await Picker.launchCameraAsync(options) : await Picker.launchImageLibraryAsync(options);
      if (photo.canceled) return;
      const asset = photo.assets[0];
      if ((asset.fileSize || 0) > 25 * 1024 * 1024) throw new Error('Choose a photo smaller than 25 MB.');
      const actions =
        Math.max(asset.width, asset.height) > 1800
          ? [{ resize: asset.width > asset.height ? { width: 1800 } : { height: 1800 } }]
          : [];
      const normalized = await Manipulator.manipulateAsync(actions.length ? asset.uri : asset.uri, actions, {
        compress: 0.9,
        format: Manipulator.SaveFormat.JPEG,
      });
      setUri(normalized.uri);
      setResult(null);
    });
  }

  async function scan() {
    if (!uri || busyLock.current) return;
    busyLock.current = true;
    setBusy(true);
    setError('');
    setScreen('analyzing');
    try {
      const data = await analyze(uri);
      setResult(data);
      setScreen('detected');
    } catch (e: any) {
      setError(e?.message || 'Could not analyze this image.');
      setScreen('scan');
    } finally {
      busyLock.current = false;
      setBusy(false);
    }
  }

  async function checkServer(showMessage = true) {
    setServerState('checking');
    try {
      const data = await api<any>('/health');
      setHealth(data);
      setServerState('online');
      if (showMessage) setMessage('Backend connected successfully.');
    } catch (e: any) {
      setHealth(null);
      setServerState('offline');
      if (showMessage) setError(e?.message || 'Backend is offline.');
    }
  }

  async function saveServer() {
    await run(async () => {
      setURL(serverURL);
      await AsyncStorage.setItem('nutrisight-server', serverURL.replace(/\/$/, ''));
      setServerURL(serverURL.replace(/\/$/, ''));
      await checkServer(true);
    });
  }

  async function changeTheme(next: 'system' | 'light' | 'dark') {
    setMode(next);
    await AsyncStorage.setItem('nutrisight-theme', next);
  }

  const logout = () =>
    run(async () => {
      try {
        await api('/api/auth/logout', 'POST', {});
      } catch {}
      await storeToken('');
      setUser(null);
      setResult(null);
      setHistory([]);
      setUri('');
      setEmail('');
      setPassword('');
      setScreen('welcome');
    });

  const confirmDeleteAccount = () => {
    const action = () =>
      run(async () => {
        await api('/api/me', 'DELETE');
        await storeToken('');
        setUser(null);
        setResult(null);
        setHistory([]);
        setUri('');
        setScreen('welcome');
      });
    if (Platform.OS === 'web') {
      if (window.confirm('Delete your NutriSight account and private history?')) action();
    } else {
      Alert.alert('Delete account?', 'Your account and saved meal history will be removed.', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: action },
      ]);
    }
  };

  const confirmDeleteScan = (scanId: string) => {
    const action = () =>
      run(async () => {
        await api(`/api/history/${scanId}`, 'DELETE');
        setHistory((items) => items.filter((item) => item.id !== scanId));
      });
    if (Platform.OS === 'web') {
      if (window.confirm('Delete this scan?')) action();
    } else {
      Alert.alert('Delete scan?', 'This scan will be removed from your history.', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: action },
      ]);
    }
  };

  if (!ready || (!fonts && !fontError)) {
    return (
      <Theme.Provider value={c}>
        <View style={{ flex: 1, backgroundColor: c.bg, alignItems: 'center', justifyContent: 'center', gap: 18 }}>
          <LogoMark size={74} />
          <T bold size={22}>NutriSight</T>
          <Motion scan size={92} />
        </View>
      </Theme.Provider>
    );
  }

  const showTabs = ['home', 'scan', 'history', 'profile', 'settings'].includes(screen);

  return (
    <Theme.Provider value={c}>
      <AppContext.Provider value={{ c }}>
        <SafeAreaView style={{ flex: 1, backgroundColor: c.bg }}>
          <StatusBar style={c === dark ? 'light' : 'dark'} />
          <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
            <View style={{ width: '100%', maxWidth: contentWidth, alignSelf: 'center', flex: 1 }}>
              <ScrollView
                ref={scrollRef}
                keyboardShouldPersistTaps="handled"
                contentContainerStyle={{
                  paddingHorizontal: compact ? 16 : 20,
                  paddingTop: 12,
                  paddingBottom: showTabs ? 120 : 36,
                  gap: 18,
                }}
                showsVerticalScrollIndicator={false}
              >
                {error ? <Notice kind="error" title="Something needs attention" body={error} /> : null}
                {message ? <Notice kind="success" title="All set" body={message} /> : null}

                {screen === 'welcome' && <Welcome go={go} />}
                {screen === 'intro' && <Intro go={go} />}
                {screen === 'login' && (
                  <AuthScreen
                    kind="login"
                    email={email}
                    password={password}
                    name={name}
                    confirmPassword={confirmPassword}
                    setEmail={setEmail}
                    setPassword={setPassword}
                    setName={setName}
                    setConfirmPassword={setConfirmPassword}
                    busy={busy}
                    onSubmit={() => auth('login')}
                    go={go}
                  />
                )}
                {screen === 'signup' && (
                  <AuthScreen
                    kind="signup"
                    email={email}
                    password={password}
                    name={name}
                    confirmPassword={confirmPassword}
                    setEmail={setEmail}
                    setPassword={setPassword}
                    setName={setName}
                    setConfirmPassword={setConfirmPassword}
                    busy={busy}
                    onSubmit={() => auth('signup')}
                    go={go}
                  />
                )}
                {screen === 'onboarding' && user && (
                  <Onboarding
                    draft={draft}
                    setDraft={setDraft}
                    busy={busy}
                    onSave={() => saveProfile('home')}
                    onSkip={skipOnboarding}
                  />
                )}
                {screen === 'home' && (
                  <Home
                    user={user}
                    dashboard={dashboard}
                    go={go}
                    serverState={serverState}
                  />
                )}
                {screen === 'scan' && (
                  <ScanScreen uri={uri} busy={busy} onPick={pick} onAnalyze={scan} user={user} />
                )}
                {screen === 'analyzing' && <Analyzing />}
                {screen === 'detected' && (
                  <DetectedScreen result={result} uri={uri} showBoxes={showBoxes} setShowBoxes={setShowBoxes} go={go} />
                )}
                {screen === 'result' && (
                  <ResultScreen
                    result={result}
                    uri={uri}
                    showBoxes={showBoxes}
                    setShowBoxes={setShowBoxes}
                    go={go}
                    user={user}
                  />
                )}
                {screen === 'insights' && (
                  <InsightsScreen result={result} go={go} user={user} />
                )}
                {screen === 'history' && (
                  <HistoryScreen
                    user={user}
                    rows={history}
                    more={moreHistory}
                    busy={busy}
                    go={go}
                    open={(item) => {
                      setResult(item);
                      setUri('');
                      go('result');
                    }}
                    remove={confirmDeleteScan}
                    loadMore={() =>
                      run(async () => {
                        const rows = await api<Result[]>(`/api/history?offset=${history.length}`);
                        setHistory((old) => [...old, ...rows]);
                        setMoreHistory(rows.length === 50);
                      })
                    }
                  />
                )}
                {screen === 'profile' && (
                  <ProfileScreen
                    user={user}
                    draft={draft}
                    setDraft={setDraft}
                    busy={busy}
                    onSave={() => saveProfile('profile')}
                    go={go}
                  />
                )}
                {screen === 'settings' && (
                  <SettingsScreen
                    user={user}
                    mode={mode}
                    onTheme={changeTheme}
                    serverURL={serverURL}
                    setServerURL={setServerURL}
                    serverState={serverState}
                    health={health}
                    saveServer={saveServer}
                    checkServer={() => checkServer(true)}
                    currentPassword={currentPassword}
                    newPassword={newPassword}
                    setCurrentPassword={setCurrentPassword}
                    setNewPassword={setNewPassword}
                    busy={busy}
                    changePassword={() =>
                      run(async () => {
                        if (newPassword.length < 8) throw new Error('New password must be at least 8 characters.');
                        const data = await api<{ token: string; user: User }>('/api/auth/password', 'PUT', {
                          current_password: currentPassword,
                          new_password: newPassword,
                        });
                        await storeToken(data.token);
                        acceptUser(data.user);
                        setCurrentPassword('');
                        setNewPassword('');
                        setMessage('Password updated.');
                      })
                    }
                    logout={logout}
                    deleteAccount={confirmDeleteAccount}
                    go={go}
                  />
                )}
              </ScrollView>
              {showTabs && <BottomTabs screen={screen} go={go} user={user} />}
            </View>
          </KeyboardAvoidingView>
        </SafeAreaView>
      </AppContext.Provider>
    </Theme.Provider>
  );
}

function ScreenHeader({ title, onBack, right }: { title: string; onBack?: () => void; right?: React.ReactNode }) {
  const c = useContext(Theme);
  return (
    <View style={{ minHeight: 48, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
      <View style={{ width: 42 }}>
        {onBack ? (
          <Pressable onPress={onBack} hitSlop={10} style={{ width: 38, height: 38, borderRadius: 13, alignItems: 'center', justifyContent: 'center' }}>
            <Icon name="arrow-back" size={21} color={c.text} />
          </Pressable>
        ) : null}
      </View>
      <T size={15} bold>{title}</T>
      <View style={{ width: 42, alignItems: 'flex-end' }}>{right}</View>
    </View>
  );
}

function BottomTabs({ screen, go, user }: { screen: Screen; go: (s: Screen) => void; user: User | null }) {
  const c = useContext(Theme);
  const items: { screen: Screen; icon: string; activeIcon: string; label: string }[] = [
    { screen: 'home', icon: 'home-outline', activeIcon: 'home', label: 'Home' },
    { screen: 'history', icon: 'time-outline', activeIcon: 'time', label: 'History' },
    { screen: 'scan', icon: 'scan-outline', activeIcon: 'scan', label: 'Scan' },
    { screen: 'profile', icon: 'person-outline', activeIcon: 'person', label: user ? 'Profile' : 'Sign in' },
    { screen: 'settings', icon: 'settings-outline', activeIcon: 'settings', label: 'Settings' },
  ];
  return (
    <View style={{ position: 'absolute', left: 0, right: 0, bottom: 0, minHeight: 75, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-around', paddingHorizontal: 8, paddingTop: 9, paddingBottom: 13, backgroundColor: c.card, borderTopWidth: 1, borderTopColor: c.line, shadowColor: c.shadow, shadowOpacity: 0.10, shadowRadius: 14, shadowOffset: { width: 0, height: -4 }, elevation: 9 }}>
      {items.map((item) => {
        const selected = screen === item.screen;
        const scan = item.screen === 'scan';
        return (
          <Pressable
            key={item.screen}
            onPress={() => go(item.screen === 'profile' && !user ? 'login' : item.screen)}
            style={{ width: 64, alignItems: 'center', justifyContent: 'center', gap: 3 }}
          >
            {scan ? (
              <View style={{ width: 51, height: 51, marginTop: -30, borderRadius: 26, backgroundColor: c.primaryDeep, alignItems: 'center', justifyContent: 'center', shadowColor: c.primaryDeep, shadowOpacity: 0.22, shadowRadius: 10, shadowOffset: { width: 0, height: 5 }, elevation: 5 }}>
                <Icon name="scan" size={25} color="#FFFFFF" />
              </View>
            ) : (
              <>
                <Icon name={selected ? item.activeIcon : item.icon} size={20} color={selected ? c.primaryDeep : c.muted} />
                <T size={9} bold color={selected ? c.primaryDeep : c.muted}>{item.label}</T>
              </>
            )}
          </Pressable>
        );
      })}
    </View>
  );
}

function Notice({ kind, title, body }: { kind: 'error' | 'success'; title: string; body: string }) {
  const c = useContext(Theme);
  const color = kind === 'error' ? c.danger : c.primaryDeep;
  return (
    <View style={{ borderWidth: 1, borderColor: kind === 'error' ? '#F0C6C6' : '#D5E8CF', backgroundColor: kind === 'error' ? '#FFF4F4' : '#F2F8EE', borderRadius: 14, padding: 13, flexDirection: 'row', gap: 10, alignItems: 'flex-start' }}>
      <Icon name={kind === 'error' ? 'alert-circle-outline' : 'checkmark-circle-outline'} color={color} size={19} />
      <View style={{ flex: 1, gap: 2 }}>
        <T bold size={12} color={color}>{title}</T>
        <T size={11}>{body}</T>
      </View>
    </View>
  );
}

function SectionHeader({ eyebrow, title, sub }: { eyebrow?: string; title: string; sub?: string }) {
  const c = useContext(Theme);
  return (
    <View style={{ gap: 4 }}>
      {eyebrow ? <T size={9} bold color={c.primary} style={{ letterSpacing: 1.3 }}>{eyebrow.toUpperCase()}</T> : null}
      <T size={24} bold>{title}</T>
      {sub ? <T size={12} color={c.muted}>{sub}</T> : null}
    </View>
  );
}

function Welcome({ go }: { go: (s: Screen) => void }) {
  const c = useContext(Theme);
  useEffect(() => {
    const timer = setTimeout(() => go('intro'), 1600);
    return () => clearTimeout(timer);
  }, []);
  return (
    <Pressable onPress={() => go('intro')} style={{ minHeight: 700, flex: 1 }}>
      <View style={{ minHeight: 700, justifyContent: 'space-between', alignItems: 'center', paddingTop: 70, overflow: 'hidden' }}>
        <View style={{ alignItems: 'center', gap: 8 }}>
          <Brand large centered tagline />
        </View>
        <View style={{ alignItems: 'center', gap: 2, marginTop: 10 }}>
          <T size={18} bold color={c.primaryDeep} style={{ fontStyle: 'italic', textAlign: 'center' }}>Healthy Choices</T>
          <T size={20} bold color={c.primaryDeep} style={{ fontStyle: 'italic', textAlign: 'center' }}>Brighter Tomorrow</T>
        </View>
        <View style={{ width: '120%', alignItems: 'center', marginBottom: -25 }}>
          <SplashMealArt size={380} />
        </View>
      </View>
    </Pressable>
  );
}

function Intro({ go }: { go: (s: Screen) => void }) {
  const c = useContext(Theme);
  return (
    <View style={{ minHeight: 700, justifyContent: 'space-between', paddingTop: 34, paddingBottom: 12 }}>
      <View style={{ alignItems: 'center', gap: 8 }}>
        <T size={24} bold style={{ textAlign: 'center' }}>Smart Nutrition</T>
        <T size={24} bold style={{ textAlign: 'center', marginTop: -7 }}>Starts Here</T>
        <T size={12} color={c.muted} style={{ textAlign: 'center', maxWidth: 280, marginTop: 4 }}>Take a photo, get instant nutrition facts and make healthier choices every day.</T>
      </View>
      <View style={{ alignItems: 'center', marginVertical: 6 }}>
        <FoodPhoneArt size={310} />
      </View>
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 18 }}>
        <View style={{ width: 56 }} />
        <View style={{ flexDirection: 'row', gap: 6 }}>
          {[0, 1, 2, 3].map((dot) => <View key={dot} style={{ width: dot === 0 ? 7 : 6, height: dot === 0 ? 7 : 6, borderRadius: 99, backgroundColor: dot === 0 ? c.primaryDeep : '#D6D8D3' }} />)}
        </View>
        <Pressable onPress={() => go('login')} style={{ width: 48, height: 48, borderRadius: 24, backgroundColor: c.primaryDeep, alignItems: 'center', justifyContent: 'center', shadowColor: c.primaryDeep, shadowOpacity: 0.2, shadowRadius: 10, shadowOffset: { width: 0, height: 5 }, elevation: 3 }}>
          <Icon name="arrow-forward" color="#FFFFFF" size={23} />
        </Pressable>
      </View>
    </View>
  );
}

function SocialButton({ icon, onPress }: { icon: string; onPress: () => void }) {
  const c = useContext(Theme);
  return (
    <Pressable onPress={onPress} style={{ width: 60, height: 48, borderRadius: 24, borderWidth: 1, borderColor: c.line, backgroundColor: c.card, alignItems: 'center', justifyContent: 'center', shadowColor: c.shadow, shadowOpacity: 0.04, shadowRadius: 7, shadowOffset: { width: 0, height: 3 } }}>
      <Icon name={icon} size={21} color={icon === 'logo-google' ? '#4285F4' : c.text} />
    </Pressable>
  );
}

function AuthScreen(props: {
  kind: 'login' | 'signup';
  email: string;
  password: string;
  name: string;
  confirmPassword: string;
  setEmail: (s: string) => void;
  setPassword: (s: string) => void;
  setName: (s: string) => void;
  setConfirmPassword: (s: string) => void;
  busy: boolean;
  onSubmit: () => void;
  go: (s: Screen) => void;
}) {
  const c = useContext(Theme);
  const signup = props.kind === 'signup';
  const socialUnavailable = () => Alert.alert('Not connected yet', 'Google and Apple sign-in need OAuth credentials in the backend. Email/password sign-in is fully connected.');
  return (
    <View style={{ minHeight: 700, paddingTop: 24, gap: 14 }}>
      <View style={{ alignItems: 'center', gap: 4 }}>
        <Brand centered />
        <T size={22} bold style={{ marginTop: 6 }}>{signup ? 'Create Account' : 'Welcome Back'}</T>
        <T size={11} color={c.muted}>{signup ? 'Join NutriSight for a healthier you' : 'Sign in to continue your healthy journey'}</T>
      </View>
      <View style={{ gap: 14, marginTop: 6 }}>
        {signup ? <Field label="Full Name" value={props.name} onChangeText={props.setName} placeholder="Enter your full name" icon="person-outline" autoComplete="name" /> : null}
        <Field label="Email" value={props.email} onChangeText={props.setEmail} placeholder="Enter your email" keyboardType="email-address" icon="mail-outline" autoComplete="email" />
        <Field label="Password" value={props.password} onChangeText={props.setPassword} placeholder={signup ? 'Create a password' : 'Enter your password'} secure icon="lock-closed-outline" autoComplete={signup ? 'new-password' : 'current-password'} />
        {!signup ? (
          <Pressable onPress={() => Alert.alert('Password reset', 'Password reset email is not connected to the current backend yet.')} style={{ alignSelf: 'flex-end', marginTop: -6 }}>
            <T size={10} bold color={c.primary}>Forget Password?</T>
          </Pressable>
        ) : null}
        <Button title={props.busy ? 'Please wait…' : signup ? 'Sign Up' : 'Sign In'} disabled={props.busy} onPress={props.onSubmit} />
      </View>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
        <View style={{ flex: 1, height: 1, backgroundColor: c.line }} />
        <T size={10} color={c.muted}>or continue with</T>
        <View style={{ flex: 1, height: 1, backgroundColor: c.line }} />
      </View>
      <View style={{ flexDirection: 'row', justifyContent: 'center', gap: 22 }}>
        <SocialButton icon="logo-google" onPress={socialUnavailable} />
        <SocialButton icon="logo-apple" onPress={socialUnavailable} />
      </View>
      <View style={{ flexDirection: 'row', justifyContent: 'center', gap: 4, marginTop: 4 }}>
        <T size={10} color={c.muted}>{signup ? 'Already have an account?' : "Don't have an account?"}</T>
        <Pressable onPress={() => props.go(signup ? 'login' : 'signup')}><T size={10} bold color={c.primary}>{signup ? 'Sign In' : 'Sign Up'}</T></Pressable>
      </View>
    </View>
  );
}

function Onboarding({ draft, setDraft, busy, onSave, onSkip }: { draft: DraftProfile; setDraft: (d: DraftProfile) => void; busy: boolean; onSave: () => void; onSkip: () => void }) {
  const c = useContext(Theme);
  const diets = ['Vegetarian', 'Vegan', 'No pork', 'No beef', 'Halal'];
  return (
    <View style={{ gap: 16 }}>
      <View style={{ alignItems: 'center', gap: 4, paddingTop: 16 }}><Brand centered /><T size={21} bold>Tell us about you</T><T size={11} color={c.muted} style={{ textAlign: 'center' }}>This helps personalize your meal insights. You can skip it.</T></View>
      <Card>
        <T bold size={15}>Your profile</T>
        <Field label="Name" value={draft.name} onChangeText={(v: string) => setDraft({ ...draft, name: v })} icon="person-outline" />
        <View style={{ flexDirection: 'row', gap: 10 }}>
          <View style={{ flex: 1 }}><Field label="Age" value={draft.age} onChangeText={(v: string) => setDraft({ ...draft, age: v })} keyboardType="number-pad" placeholder="Optional" /></View>
          <View style={{ flex: 1 }}><Field label="Sex" value={draft.sex} onChangeText={(v: string) => setDraft({ ...draft, sex: v })} placeholder="Optional" /></View>
        </View>
        <View style={{ flexDirection: 'row', gap: 10 }}>
          <View style={{ flex: 1 }}><Field label="Height cm" value={draft.height} onChangeText={(v: string) => setDraft({ ...draft, height: v })} keyboardType="decimal-pad" placeholder="Optional" /></View>
          <View style={{ flex: 1 }}><Field label="Weight kg" value={draft.weight} onChangeText={(v: string) => setDraft({ ...draft, weight: v })} keyboardType="decimal-pad" placeholder="Optional" /></View>
        </View>
      </Card>
      <Card>
        <T bold size={15}>Dietary preferences</T>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 7 }}>
          {diets.map((diet) => {
            const selected = draft.preferences.includes(diet);
            return <Chip key={diet} label={diet} selected={selected} onPress={() => setDraft({ ...draft, preferences: selected ? draft.preferences.filter((x) => x !== diet) : [...draft.preferences, diet] })} />;
          })}
        </View>
        <Field label="Allergies" value={draft.allergies} onChangeText={(v: string) => setDraft({ ...draft, allergies: v })} placeholder="e.g. milk, peanut, egg" icon="medical-outline" />
        <Field label="Foods to avoid" value={draft.avoid} onChangeText={(v: string) => setDraft({ ...draft, avoid: v })} placeholder="e.g. shrimp, pork" icon="close-circle-outline" />
      </Card>
      <Card>
        <T bold size={15}>My goal</T>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 7 }}>
          {['Build mindful habits', 'Maintain weight', 'Eat more protein', 'Improve meal balance'].map((goal) => <Chip key={goal} label={goal} selected={draft.goal === goal} onPress={() => setDraft({ ...draft, goal })} />)}
        </View>
      </Card>
      <Button title={busy ? 'Saving…' : 'Save & Continue'} disabled={busy} onPress={onSave} />
      <Pressable disabled={busy} onPress={onSkip}><T size={12} color={c.muted} style={{ textAlign: 'center' }}>Skip for now</T></Pressable>
    </View>
  );
}

function Chip({ label, selected, onPress, tone = 'green' }: { label: string; selected: boolean; onPress: () => void; tone?: 'green' | 'red' | 'blue' }) {
  const c = useContext(Theme);
  const bg = tone === 'red' ? '#FCE7E4' : tone === 'blue' ? '#EEF0FA' : c.tint;
  const fg = tone === 'red' ? '#B84B49' : tone === 'blue' ? '#4E5B8B' : c.primaryDeep;
  return (
    <Pressable onPress={onPress} style={{ paddingHorizontal: 11, paddingVertical: 7, borderRadius: 99, borderWidth: 1, borderColor: selected ? fg : c.line, backgroundColor: selected ? bg : c.card, flexDirection: 'row', gap: 4, alignItems: 'center' }}>
      {selected ? <Icon name="checkmark" size={12} color={fg} /> : null}
      <T size={9} bold color={selected ? fg : c.text}>{label}</T>
    </Pressable>
  );
}

function Home({ user, dashboard, go, serverState }: { user: User | null; dashboard: Dashboard; go: (s: Screen) => void; serverState: 'checking' | 'online' | 'offline' }) {
  const c = useContext(Theme);
  const firstName = user?.name?.split(' ')[0] || 'there';
  return (
    <View style={{ gap: 16 }}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        <Brand />
        <ServerBadge status={serverState} />
      </View>
      <View style={{ gap: 3 }}><T size={22} bold>Hi, {firstName}</T><T size={11} color={c.muted}>Ready to understand your next meal?</T></View>
      <Card style={{ backgroundColor: c.tint, borderColor: '#DDEACF', overflow: 'hidden', padding: 18 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
          <View style={{ flex: 1, gap: 6 }}><T size={10} bold color={c.primary}>AI FOOD DETECTION</T><T size={20} bold>Scan your plate in seconds</T><T size={11} color={c.muted}>YOLO detection, ViT-LSTM refinement and whole-meal nutrition estimation.</T></View>
          <FoodPhoneArt size={138} />
        </View>
        <Button title="Scan Food" icon="scan-outline" onPress={() => go('scan')} />
      </Card>
      <View style={{ flexDirection: 'row', gap: 9 }}>
        <MetricCard icon="restaurant-outline" value={user ? String(dashboard.total_scans) : '—'} label={user ? 'Meals scanned' : 'Guest mode'} />
        <MetricCard icon="sparkles-outline" value="3" label="AI stages" />
        <MetricCard icon="shield-checkmark-outline" value="Private" label="History" />
      </View>
      {user && dashboard.recent.length > 0 ? (
        <View style={{ gap: 9 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}><T size={15} bold>Recent meals</T><Pressable onPress={() => go('history')}><T size={10} bold color={c.primary}>View all</T></Pressable></View>
          {dashboard.recent.slice(0, 3).map((item) => <MealRow key={item.id} result={item} onPress={() => go('history')} />)}
        </View>
      ) : (
        <Card><View style={{ flexDirection: 'row', alignItems: 'center', gap: 11 }}><View style={{ width: 42, height: 42, borderRadius: 13, backgroundColor: c.tint, alignItems: 'center', justifyContent: 'center' }}><Icon name="time-outline" size={20} /></View><View style={{ flex: 1 }}><T size={12} bold>{user ? 'Your food history starts here' : 'Save your meal history'}</T><T size={10} color={c.muted}>{user ? 'Your first scan will appear here.' : 'Sign up to save scans and preferences.'}</T></View></View>{!user ? <Button secondary compact title="Create account" onPress={() => go('signup')} /> : null}</Card>
      )}
    </View>
  );
}

function MetricCard({ icon, value, label }: { icon: string; value: string; label: string }) {
  const c = useContext(Theme);
  return <View style={{ flex: 1, minHeight: 86, borderRadius: 15, padding: 11, backgroundColor: c.card, borderWidth: 1, borderColor: c.line, gap: 4 }}><Icon name={icon} size={18} /><T size={14} bold numberOfLines={1}>{value}</T><T size={9} color={c.muted}>{label}</T></View>;
}

function ScanScreen({ uri, busy, onPick, onAnalyze, user }: { uri: string; busy: boolean; onPick: (camera: boolean) => void; onAnalyze: () => void; user: User | null }) {
  const c = useContext(Theme);
  return (
    <View style={{ gap: 16 }}>
      <ScreenHeader title="Scan Food" onBack={() => router.replace('/home' as any)} />
      {uri ? (
        <>
          <Card style={{ padding: 7 }}><Image source={{ uri }} resizeMode="contain" style={{ width: '100%', height: 315, borderRadius: 12, backgroundColor: c.elevated }} /></Card>
          <Button title={busy ? 'Analyzing…' : 'Analyze Food'} icon="sparkles-outline" disabled={busy} onPress={onAnalyze} />
          <View style={{ flexDirection: 'row', gap: 9 }}><View style={{ flex: 1 }}><Button secondary compact title="Retake" icon="camera-outline" disabled={busy} onPress={() => onPick(true)} /></View><View style={{ flex: 1 }}><Button secondary compact title="Choose another" icon="images-outline" disabled={busy} onPress={() => onPick(false)} /></View></View>
        </>
      ) : (
        <>
          <ScanChoice icon="camera-outline" title="Take a Photo" sub="Using camera" onPress={() => onPick(true)} />
          <ScanChoice icon="image-outline" title="Choose from Gallery" sub="Select an image" onPress={() => onPick(false)} />
        </>
      )}
      <View style={{ marginTop: 8, gap: 12 }}>
        <T size={14} bold>Tips for better results</T>
        {['Use good lighting', 'Keep food in focus', 'Try to capture the whole plate', 'Multiple food items are supported'].map((tip) => (
          <View key={tip} style={{ flexDirection: 'row', gap: 9, alignItems: 'center' }}><View style={{ width: 19, height: 19, borderRadius: 10, backgroundColor: '#3E9D52', alignItems: 'center', justifyContent: 'center' }}><Icon name="checkmark" size={13} color="#FFFFFF" /></View><T size={11} color={c.muted}>{tip}</T></View>
        ))}
      </View>
      <T size={9} color={c.muted}>Photos are sent to your configured backend for analysis. {user ? 'Results are saved to your private history.' : 'Guest results are not saved.'}</T>
    </View>
  );
}

function ScanChoice({ icon, title, sub, onPress }: { icon: string; title: string; sub: string; onPress: () => void }) {
  const c = useContext(Theme);
  return (
    <Pressable onPress={onPress} style={({ pressed }) => ({ minHeight: 118, borderRadius: 16, backgroundColor: c.card, borderWidth: 1, borderColor: c.line, alignItems: 'center', justifyContent: 'center', gap: 6, opacity: pressed ? 0.82 : 1, shadowColor: c.shadow, shadowOpacity: 0.06, shadowRadius: 12, shadowOffset: { width: 0, height: 5 }, elevation: 1 })}>
      <View style={{ width: 45, height: 45, borderRadius: 15, backgroundColor: c.tint, alignItems: 'center', justifyContent: 'center' }}><Icon name={icon} size={25} /></View>
      <T size={13} bold>{title}</T>
      <T size={10} color={c.muted}>{sub}</T>
    </Pressable>
  );
}

function Analyzing() {
  const c = useContext(Theme);
  const steps = ['Detecting food items', 'Recognizing with ViT-LSTM', 'Estimating nutrition', 'Preparing results'];
  return (
    <View style={{ minHeight: 700, gap: 20 }}>
      <ScreenHeader title="Analyzing Food" />
      <View style={{ alignItems: 'center', gap: 13, marginTop: 14 }}>
        <AnalyzeArt size={235} />
        <T size={12} bold>Detecting food items...</T>
        <View style={{ width: '86%', height: 8, borderRadius: 99, backgroundColor: '#E5E6E1', overflow: 'hidden' }}><View style={{ width: '68%', height: '100%', borderRadius: 99, backgroundColor: c.primaryDeep }} /></View>
        <T size={11} bold>68%</T>
      </View>
      <Card style={{ gap: 14 }}>
        {steps.map((label, index) => (
          <View key={label} style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
            <View style={{ width: 19, height: 19, borderRadius: 10, borderWidth: index === 0 ? 0 : 1.3, borderColor: '#ABB7B0', backgroundColor: index === 0 ? c.primaryDeep : 'transparent', alignItems: 'center', justifyContent: 'center' }}>{index === 0 ? <Icon name="checkmark" size={13} color="#FFFFFF" /> : null}</View>
            <T size={11} color={index === 0 ? c.text : c.muted}>{label}</T>
          </View>
        ))}
      </Card>
    </View>
  );
}

function ImageWithBoxes({ uri, result, showBoxes, height = 330 }: { uri: string; result: Result; showBoxes: boolean; height?: number }) {
  const c = useContext(Theme);
  const colors = ['#4CA64F', '#2F8BE6', '#E53E3E', '#E7A523', '#8B5CF6'];
  return (
    <View style={{ width: '100%', height, overflow: 'hidden', backgroundColor: c.elevated }}>
      <Image source={{ uri }} resizeMode="cover" style={{ width: '100%', height: '100%' }} />
      {showBoxes ? result.foods.map((food, index) => {
        const color = colors[index % colors.length];
        return (
          <View key={`${food.name}-${index}`} style={{ position: 'absolute', left: `${food.bbox[0] * 100}%`, top: `${food.bbox[1] * 100}%`, width: `${(food.bbox[2] - food.bbox[0]) * 100}%`, height: `${(food.bbox[3] - food.bbox[1]) * 100}%`, borderWidth: 2, borderColor: color }}>
            <Text style={{ alignSelf: 'flex-start', backgroundColor: color, color: '#FFFFFF', paddingHorizontal: 5, paddingVertical: 2, fontSize: 10, fontFamily: 'Quicksand_700Bold' }}>{nice(food.name)} {Math.round(food.confidence * 100)}%</Text>
          </View>
        );
      }) : null}
    </View>
  );
}

function FoodThumb({ uri, food, w, h, radius = 10 }: { uri?: string; food: Result['foods'][number]; w: number; h: number; radius?: number }) {
  const c = useContext(Theme);
  const [x1, y1, x2, y2] = food.bbox || [0, 0, 1, 1];
  const bw = Math.max(0.05, x2 - x1), bh = Math.max(0.05, y2 - y1);
  return (
    <View style={{ width: w, height: h, borderRadius: radius, overflow: 'hidden', backgroundColor: c.tint, alignItems: 'center', justifyContent: 'center' }}>
      {uri ? <Image source={{ uri }} resizeMode="stretch" style={{ position: 'absolute', width: w / bw, height: h / bh, left: -x1 * (w / bw), top: -y1 * (h / bh) }} /> : <Icon name="restaurant" size={Math.min(w, h) * 0.45} />}
    </View>
  );
}

function DetectedScreen({ result, uri, showBoxes, setShowBoxes, go }: { result: Result | null; uri: string; showBoxes: boolean; setShowBoxes: (v: boolean) => void; go: (s: Screen) => void }) {
  const c = useContext(Theme);
  if (!result) return <EmptyState icon="scan-outline" title="No result yet" body="Scan a meal first." action="Scan Food" onPress={() => go('scan')} />;
  return (
    <View style={{ gap: 13 }}>
      <ScreenHeader
        title="Detected Items"
        onBack={() => go('scan')}
        right={<Pressable onPress={() => setShowBoxes(!showBoxes)} style={{ width: 34, height: 34, borderRadius: 12, borderWidth: 1, borderColor: c.line, alignItems: 'center', justifyContent: 'center' }}><Icon name={showBoxes ? 'eye-off-outline' : 'eye-outline'} size={17} /></Pressable>}
      />
      {uri ? <View style={{ borderRadius: 14, overflow: 'hidden', borderWidth: 1, borderColor: c.line }}><ImageWithBoxes uri={uri} result={result} showBoxes={showBoxes} height={350} /></View> : null}
      <View style={{ flexDirection: 'row', gap: 8 }}>
        {result.foods.slice(0, 4).map((food, index) => (
          <View key={`${food.name}-${index}`} style={{ flex: 1, alignItems: 'center', gap: 4 }}>
            <FoodThumb uri={uri} food={food} w={74} h={62} />
            <T size={9} bold numberOfLines={1} style={{ textAlign: 'center' }}>{nice(food.name)}</T>
          </View>
        ))}
      </View>
      <Button title="View Nutrition Results" icon="arrow-forward" onPress={() => go('result')} />
    </View>
  );
}

function ResultScreen({ result, uri, go, user }: { result: Result | null; uri: string; showBoxes: boolean; setShowBoxes: (v: boolean) => void; go: (s: Screen) => void; user: User | null }) {
  const c = useContext(Theme);
  if (!result) return <EmptyState icon="nutrition-outline" title="No nutrition result" body="Scan a meal to see results." action="Scan Food" onPress={() => go('scan')} />;
  const n = result.nutrition;
  return (
    <View style={{ gap: 15 }}>
      <ScreenHeader title="Nutrition Results" onBack={() => go('detected')} />
      {n ? (
        <>
          <View style={{ gap: 7 }}><T size={13} bold>Total Nutrition (Entire Meal)</T><View style={{ flexDirection: 'row', gap: 7 }}>
            <NutritionMetric label="Calories" value={`${Math.round(n.calories || 0)}`} unit="" color="#EA6A31" />
            <NutritionMetric label="Protein" value={`${Math.round(n.protein || 0)}g`} unit="" color={c.primaryDeep} />
            <NutritionMetric label="Carbs" value={`${Math.round(n.carbs || 0)}g`} unit="" color={c.blue} />
            <NutritionMetric label="Fat" value={`${Math.round(n.fat || 0)}g`} unit="" color={c.orange} />
          </View></View>
        </>
      ) : <Notice kind="error" title="No nutrition estimate" body="No food was detected strongly enough to estimate the meal." />}
      <View style={{ gap: 8 }}>
        <T size={13} bold>Detected Items</T>
        {result.foods.length ? result.foods.map((food, index) => <FoodResultRow key={`${food.name}-${index}`} food={food} index={index} uri={uri} />) : <EmptyState icon="search-outline" title="No food detected" body="Try a brighter, sharper image with the whole plate visible." />}
      </View>
      <Button title="View Nutrition Insights" icon="heart-outline" onPress={() => go('insights')} />
      <Button secondary title="Scan Another Meal" icon="scan-outline" onPress={() => go('scan')} />
      <T size={9} color={c.muted} style={{ textAlign: 'center' }}>{result.saved ? 'Saved to your private NutriSight history.' : user ? 'This result was not saved.' : 'Guest result. Sign in to save your history.'}</T>
    </View>
  );
}

function NutritionMetric({ label, value, unit, color }: { label: string; value: string; unit: string; color?: string }) {
  const c = useContext(Theme);
  return <View style={{ flex: 1, minHeight: 68, borderRadius: 12, paddingHorizontal: 5, paddingVertical: 9, backgroundColor: c.card, borderWidth: 1, borderColor: c.line, alignItems: 'center', justifyContent: 'center', gap: 2 }}><T size={16} bold color={color || c.text}>{value}{unit}</T><T size={8} color={c.muted}>{label}</T></View>;
}

function FoodResultRow({ food, index, uri }: { food: Result['foods'][number]; index: number; uri?: string }) {
  const c = useContext(Theme);
  const n: any = (food as any).nutrition;
  return (
    <Card style={{ padding: 10 }}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 11 }}>
        <FoodThumb uri={uri} food={food} w={50} h={50} radius={12} />
        <View style={{ flex: 1, gap: 2 }}>
          <T size={12} bold>{nice(food.name)}</T>
          <T size={9} color={c.muted}>{n?.grams ? `${Math.round(n.grams)}g · ` : ''}{Math.round(food.confidence * 100)}% confidence</T>
          {n ? <T size={9} color={c.muted}>{Math.round(n.protein || 0)}g P   {Math.round(n.carbs || 0)}g C   {Math.round(n.fat || 0)}g F</T> : null}
        </View>
        {n?.calories ? <T size={13} bold color={c.primaryDeep}>{Math.round(n.calories)} cal</T> : <T size={11} bold color={c.primaryDeep}>#{index + 1}</T>}
      </View>
    </Card>
  );
}

function FoodRow({ food, index, compact = false }: { food: Result['foods'][number]; index: number; compact?: boolean }) {
  const c = useContext(Theme);
  return (
    <View style={{ paddingVertical: compact ? 7 : 10, borderTopWidth: index === 0 ? 0 : 1, borderTopColor: c.line, flexDirection: 'row', gap: 9, alignItems: 'center' }}>
      <View style={{ width: 31, height: 31, borderRadius: 10, backgroundColor: c.tint, alignItems: 'center', justifyContent: 'center' }}><T size={10} bold>{index + 1}</T></View>
      <View style={{ flex: 1 }}><T size={10} bold>{nice(food.name)}</T><T size={8} color={c.muted}>{food.source}</T></View>
      <T size={9} bold color={c.primary}>{Math.round(food.confidence * 100)}%</T>
    </View>
  );
}

function InsightsScreen({ result, go, user }: { result: Result | null; go: (s: Screen) => void; user: User | null }) {
  const c = useContext(Theme);
  if (!result) return <EmptyState icon="heart-outline" title="No insights yet" body="Scan a meal first." action="Scan Food" onPress={() => go('scan')} />;
  const n = result.nutrition || {};
  const protein = Number(n.protein || 0);
  const carbs = Number(n.carbs || 0);
  const fat = Number(n.fat || 0);
  const pk = protein * 4, ck = carbs * 4, fk = fat * 9, total = pk + ck + fk || 1;
  const pPct = Math.round(pk / total * 100), cPct = Math.round(ck / total * 100), fPct = Math.max(0, 100 - pPct - cPct);
  const benefits = [
    protein >= 20 ? 'Provides a meaningful amount of protein' : protein > 0 ? 'Contains protein' : null,
    carbs > 0 ? 'Contains carbohydrates, a source of food energy' : null,
    fat > 0 ? 'Includes dietary fat' : null,
  ].filter(Boolean) as string[];
  return (
    <View style={{ gap: 15 }}>
      <ScreenHeader title="Your Nutrition Insights" onBack={() => go('result')} />
      <View style={{ borderRadius: 14, padding: 13, backgroundColor: '#EAF5DF', flexDirection: 'row', gap: 10, alignItems: 'center' }}>
        <View style={{ width: 39, height: 39, borderRadius: 20, backgroundColor: c.primaryDeep, alignItems: 'center', justifyContent: 'center' }}><Icon name="leaf" size={19} color="#FFFFFF" /></View>
        <View style={{ flex: 1 }}><T size={11} bold>Great Choice! 🎉</T><T size={9} color={c.muted}>{user ? 'Your saved preferences are included where the backend provides alerts.' : 'Sign in to personalize future dietary checks.'}</T></View>
      </View>
      <Card>
        <T size={12} bold>Nutrition Balance</T>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 16 }}>
          <MacroDonut protein={protein} carbs={carbs} fat={fat} size={126} />
          <View style={{ flex: 1, gap: 9 }}>
            <LegendRow color={c.primaryDeep} label="Protein" value={`${pPct}%`} />
            <LegendRow color={c.blue} label="Carbs" value={`${cPct}%`} />
            <LegendRow color={c.orange} label="Fat" value={`${fPct}%`} />
          </View>
        </View>
      </Card>
      <View style={{ gap: 10 }}><T size={12} bold>Health Benefits</T>
        {benefits.map((item) => <View key={item} style={{ flexDirection: 'row', gap: 9, alignItems: 'flex-start' }}><View style={{ width: 19, height: 19, borderRadius: 10, backgroundColor: c.primaryDeep, alignItems: 'center', justifyContent: 'center' }}><Icon name="checkmark" size={12} color="#FFFFFF" /></View><T size={10} style={{ flex: 1 }}>{item}</T></View>)}
        {result.alerts.map((alert, index) => <View key={`${alert.title}-${index}`} style={{ flexDirection: 'row', gap: 9, alignItems: 'flex-start' }}><Icon name={alert.level === 'warning' ? 'alert-circle' : 'information-circle'} size={19} color={alert.level === 'warning' ? c.warn : c.primary} /><View style={{ flex: 1 }}><T size={10} bold>{alert.title}</T><T size={9} color={c.muted}>{alert.reason}</T></View></View>)}
      </View>
      {result.warnings.length ? <Card><T size={11} bold>Keep in mind</T>{result.warnings.map((warning, index) => <T key={index} size={9} color={c.muted}>• {warning}</T>)}</Card> : null}
      <Button secondary title="Back to Results" onPress={() => go('result')} />
    </View>
  );
}

function LegendRow({ color, label, value }: { color: string; label: string; value: string }) {
  return <View style={{ flexDirection: 'row', alignItems: 'center', gap: 7 }}><View style={{ width: 9, height: 9, borderRadius: 5, backgroundColor: color }} /><T size={10} style={{ flex: 1 }}>{label}</T><T size={10} bold>{value}</T></View>;
}

function HistoryScreen({ user, rows, more, busy, go, open, remove, loadMore }: { user: User | null; rows: Result[]; more: boolean; busy: boolean; go: (s: Screen) => void; open: (r: Result) => void; remove: (id: string) => void; loadMore: () => void }) {
  const c = useContext(Theme);
  const month = new Date().toLocaleDateString(undefined, { month: 'long', year: 'numeric' });
  return (
    <View style={{ gap: 14 }}>
      <ScreenHeader title="My Food History" right={<Icon name="calendar-outline" size={19} color={c.text} />} />
      <T size={10} bold color={c.primary}>‹ {month}</T>
      {!user ? <EmptyState icon="lock-closed-outline" title="Your private food history" body="Sign in to save and revisit meal scans." action="Sign In" onPress={() => go('login')} /> : rows.length === 0 ? <EmptyState icon="time-outline" title="Nothing here yet" body="Your first saved meal will appear here." action="Scan Food" onPress={() => go('scan')} /> : rows.map((item) => <HistoryMealRow key={item.id} result={item} onPress={() => open(item)} onDelete={() => remove(item.id)} busy={busy} />)}
      {user && more && rows.length > 0 ? <Button secondary compact title="Load more" disabled={busy} onPress={loadMore} /> : null}
    </View>
  );
}

function HistoryMealRow({ result, onPress, onDelete, busy }: { result: Result; onPress: () => void; onDelete: () => void; busy: boolean }) {
  const c = useContext(Theme);
  const title = result.foods.length ? result.foods.slice(0, 2).map((food) => nice(food.name)).join(' + ') : 'Meal Scan';
  return (
    <Pressable onPress={onPress} onLongPress={onDelete} disabled={busy}>
      <Card style={{ padding: 10 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
          <View style={{ width: 50, height: 50, borderRadius: 13, backgroundColor: '#F5EFE2', alignItems: 'center', justifyContent: 'center' }}><Icon name="restaurant" size={21} /></View>
          <View style={{ flex: 1, gap: 2 }}><T size={11} bold numberOfLines={1}>{title}</T><T size={9} color={c.muted}>{result.nutrition ? `${Math.round(result.nutrition.calories || 0)} cal` : 'Nutrition unavailable'} · {result.foods.length} item{result.foods.length === 1 ? '' : 's'}</T><T size={8} color={c.muted}>{new Date(result.created_at * 1000).toLocaleString()}</T></View>
          <Icon name="chevron-forward" size={17} color={c.muted} />
        </View>
      </Card>
    </Pressable>
  );
}

function MealRow({ result, onPress }: { result: Result; onPress: () => void }) {
  const c = useContext(Theme);
  const title = result.foods.length ? result.foods.slice(0, 2).map((food) => nice(food.name)).join(' + ') : 'Meal Scan';
  return <Pressable onPress={onPress}><Card style={{ padding: 10 }}><View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}><View style={{ width: 43, height: 43, borderRadius: 12, backgroundColor: c.tint, alignItems: 'center', justifyContent: 'center' }}><Icon name="restaurant-outline" size={19} /></View><View style={{ flex: 1, gap: 1 }}><T size={11} bold numberOfLines={1}>{title}</T><T size={9} color={c.muted}>{result.nutrition ? `${Math.round(result.nutrition.calories || 0)} cal` : 'No nutrition estimate'} · {new Date(result.created_at * 1000).toLocaleDateString()}</T></View><Icon name="chevron-forward" size={16} color={c.muted} /></View></Card></Pressable>;
}

function ProfileScreen({ user, draft, setDraft, busy, onSave, go }: { user: User | null; draft: DraftProfile; setDraft: (d: DraftProfile) => void; busy: boolean; onSave: () => void; go: (s: Screen) => void }) {
  const c = useContext(Theme);
  const [editing, setEditing] = useState(false);
  if (!user) return <View style={{ gap: 16 }}><ScreenHeader title="My Profile" /><EmptyState icon="person-circle-outline" title="Sign in to continue" body="Create an account to save dietary preferences and history." action="Sign In" onPress={() => go('login')} /></View>;
  const p = user.profile || {};
  const options = ['Vegetarian', 'Vegan', 'No pork', 'No beef', 'Halal'];
  if (editing) {
    return (
      <View style={{ gap: 14 }}>
        <ScreenHeader title="Edit Profile" onBack={() => setEditing(false)} />
        <Card>
          <Field label="Name" value={draft.name} onChangeText={(v: string) => setDraft({ ...draft, name: v })} icon="person-outline" />
          <View style={{ flexDirection: 'row', gap: 9 }}><View style={{ flex: 1 }}><Field label="Age" value={draft.age} onChangeText={(v: string) => setDraft({ ...draft, age: v })} keyboardType="number-pad" /></View><View style={{ flex: 1 }}><Field label="Sex" value={draft.sex} onChangeText={(v: string) => setDraft({ ...draft, sex: v })} /></View></View>
          <View style={{ flexDirection: 'row', gap: 9 }}><View style={{ flex: 1 }}><Field label="Height cm" value={draft.height} onChangeText={(v: string) => setDraft({ ...draft, height: v })} keyboardType="decimal-pad" /></View><View style={{ flex: 1 }}><Field label="Weight kg" value={draft.weight} onChangeText={(v: string) => setDraft({ ...draft, weight: v })} keyboardType="decimal-pad" /></View></View>
          <Field label="Allergies" value={draft.allergies} onChangeText={(v: string) => setDraft({ ...draft, allergies: v })} placeholder="Comma separated" />
          <Field label="Foods to avoid" value={draft.avoid} onChangeText={(v: string) => setDraft({ ...draft, avoid: v })} placeholder="Comma separated" />
        </Card>
        <Card><T size={12} bold>Dietary Preferences</T><View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 7 }}>{options.map((option) => <Chip key={option} label={option} selected={draft.preferences.includes(option)} onPress={() => setDraft({ ...draft, preferences: draft.preferences.includes(option) ? draft.preferences.filter((x) => x !== option) : [...draft.preferences, option] })} />)}</View></Card>
        <Card><T size={12} bold>My Goal</T><View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 7 }}>{['Build mindful habits', 'Maintain weight', 'Eat more protein', 'Improve meal balance'].map((goal) => <Chip key={goal} label={goal} selected={draft.goal === goal} onPress={() => setDraft({ ...draft, goal })} />)}</View></Card>
        <Button title={busy ? 'Saving…' : 'Save Profile'} disabled={busy} onPress={() => { onSave(); setEditing(false); }} />
      </View>
    );
  }
  const prefs = (p.preferences || []) as string[];
  return (
    <View style={{ gap: 14 }}>
      <ScreenHeader title="My Profile" right={<Pressable onPress={() => go('settings')}><Icon name="settings-outline" size={20} color={c.text} /></Pressable>} />
      <View style={{ alignItems: 'center', gap: 5 }}><AvatarArt size={72} /><T size={15} bold>{user.name}</T><T size={9} color={c.muted}>{user.email}</T></View>
      <Card style={{ padding: 12 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <View style={{ flex: 1, gap: 12 }}>
            <View style={{ flexDirection: 'row' }}><ProfileStat icon={String(p.sex||"").toLowerCase().startsWith("m") ? "male-outline" : "female-outline"} value={p.sex || '—'} /><ProfileStat icon="calendar-outline" value={p.age ? `${p.age} years` : '—'} /></View>
            <View style={{ flexDirection: 'row' }}><ProfileStat icon="swap-vertical-outline" value={p.height_cm ? `${p.height_cm} cm` : '—'} /><ProfileStat icon="speedometer-outline" value={p.weight_kg ? `${p.weight_kg} kg` : '—'} /></View>
          </View>
          <Pressable onPress={() => setEditing(true)} style={{ paddingHorizontal: 16, height: 34, borderRadius: 17, backgroundColor: c.primaryDeep, alignItems: 'center', justifyContent: 'center', alignSelf: 'flex-start' }}><T size={10} bold color="#FFFFFF">Edit</T></Pressable>
        </View>
      </Card>
      <View style={{ gap: 7 }}><T size={12} bold>My Goals</T><Card style={{ padding: 11 }}><View style={{ flexDirection: 'row', gap: 9, alignItems: 'center' }}><Icon name="body-outline" size={18} /><T size={10} style={{ flex: 1 }}>{p.goal || 'Build mindful habits'}</T><Icon name="chevron-forward" size={15} color={c.muted} /></View></Card></View>
      <View style={{ gap: 7 }}><T size={12} bold>Dietary Preferences</T><View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 7 }}>{prefs.length ? prefs.map((pref, index) => <View key={pref} style={{ paddingHorizontal: 11, paddingVertical: 7, borderRadius: 99, backgroundColor: index % 3 === 0 ? '#FCE8E4' : index % 3 === 1 ? '#EEEAF8' : '#E3F1D8' }}><T size={9} bold color={index % 3 === 0 ? '#B24B49' : index % 3 === 1 ? '#535D88' : c.primaryDeep}>{pref}</T></View>) : <T size={10} color={c.muted}>No preferences added yet</T>}</View></View>
    </View>
  );
}

function ProfileStat({ icon, value }: { icon: string; value: string | number }) {
  const c = useContext(Theme);
  return <View style={{ flex: 1, minWidth: 58, alignItems: 'center', gap: 3 }}><Icon name={icon} size={15} /><T size={8} bold numberOfLines={1}>{String(value)}</T></View>;
}

function SettingsScreen(props: {
  user: User | null;
  mode: 'system' | 'light' | 'dark';
  onTheme: (v: 'system' | 'light' | 'dark') => void;
  serverURL: string;
  setServerURL: (v: string) => void;
  serverState: 'checking' | 'online' | 'offline';
  health: any;
  saveServer: () => void;
  checkServer: () => void;
  currentPassword: string;
  newPassword: string;
  setCurrentPassword: (v: string) => void;
  setNewPassword: (v: string) => void;
  busy: boolean;
  changePassword: () => void;
  logout: () => void;
  deleteAccount: () => void;
  go: (s: Screen) => void;
}) {
  const c = useContext(Theme);
  const [notifications, setNotifications] = useState(true);
  const [showConnection, setShowConnection] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const isDark = props.mode === 'dark';
  return (
    <View style={{ gap: 13 }}>
      <ScreenHeader title="Settings" />
      <Card style={{ padding: 0, gap: 0, overflow: 'hidden' }}>
        <MenuRow icon="moon-outline" label="Dark Mode" right={<Switch value={isDark} onValueChange={(v) => props.onTheme(v ? 'dark' : 'light')} trackColor={{ false: '#D9DDDA', true: c.primary }} thumbColor="#FFFFFF" />} />
        <MenuRow icon="globe-outline" label="Language" value="English" onPress={() => Alert.alert('Language', 'English is currently available. Additional languages can be added later.')} />
        <MenuRow icon="notifications-outline" label="Notifications" right={<Switch value={notifications} onValueChange={setNotifications} trackColor={{ false: '#D9DDDA', true: c.primary }} thumbColor="#FFFFFF" />} />
        <MenuRow icon="key-outline" label="Change Password" onPress={() => setShowPassword(!showPassword)} />
        <MenuRow icon="server-outline" label="Backend Connection" value={props.serverState === 'online' ? 'Online' : props.serverState === 'offline' ? 'Offline' : 'Checking'} onPress={() => setShowConnection(!showConnection)} />
        <MenuRow icon="shield-outline" label="Privacy Policy" onPress={() => Alert.alert('Privacy', 'Meal photos are sent to your configured backend for analysis. Signed-in results are saved in your private account history.')} />
        <MenuRow icon="information-circle-outline" label="About" last onPress={() => Alert.alert('NutriSight', 'AI-assisted multi-food recognition and whole-meal nutrition estimation.')} />
      </Card>
      {showConnection ? (
        <Card>
          <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}><T size={12} bold>Backend Connection</T><ServerBadge status={props.serverState} /></View>
          <Field label="Server address" value={props.serverURL} onChangeText={props.setServerURL} placeholder="http://192.168.1.10:8000" icon="server-outline" />
          <View style={{ flexDirection: 'row', gap: 8 }}><View style={{ flex: 1 }}><Button secondary compact title="Test" disabled={props.busy} onPress={props.checkServer} /></View><View style={{ flex: 1 }}><Button compact title="Save" disabled={props.busy} onPress={props.saveServer} /></View></View>
          {props.health?.models ? <View style={{ gap: 5 }}><HealthRow label="YOLO model" ok={!!props.health.models.yolo_present} /><HealthRow label="ViT refinement" ok={!!props.health.models.classifier_present} /><HealthRow label="Nutrition model" ok={!!props.health.models.nutrition_present} /></View> : null}
        </Card>
      ) : null}
      {showPassword && props.user ? (
        <Card><Field label="Current password" value={props.currentPassword} onChangeText={props.setCurrentPassword} secure icon="lock-closed-outline" /><Field label="New password" value={props.newPassword} onChangeText={props.setNewPassword} secure icon="key-outline" /><Button secondary compact title="Change Password" disabled={props.busy} onPress={props.changePassword} /></Card>
      ) : null}
      {props.user ? <Button danger title="Logout" disabled={props.busy} onPress={props.logout} /> : <Button title="Sign In" onPress={() => props.go('login')} />}
      {props.user ? <Pressable onPress={props.deleteAccount} disabled={props.busy}><T size={9} color={c.muted} style={{ textAlign: 'center' }}>Delete account and saved history</T></Pressable> : null}
    </View>
  );
}

function MenuRow({ icon, label, value, right, onPress, last = false }: { icon: string; label: string; value?: string; right?: React.ReactNode; onPress?: () => void; last?: boolean }) {
  const c = useContext(Theme);
  return (
    <Pressable onPress={onPress} disabled={!onPress && !right} style={({ pressed }) => ({ minHeight: 54, paddingHorizontal: 13, flexDirection: 'row', alignItems: 'center', gap: 10, borderBottomWidth: last ? 0 : 1, borderBottomColor: c.line, opacity: pressed ? 0.75 : 1 })}>
      <Icon name={icon} size={17} color={c.text} />
      <T size={10} style={{ flex: 1 }}>{label}</T>
      {right || <>{value ? <T size={9} color={c.muted}>{value}</T> : null}<Icon name="chevron-forward" size={15} color={c.muted} /></>}
    </Pressable>
  );
}

function HealthRow({ label, ok }: { label: string; ok: boolean }) {
  const c = useContext(Theme);
  return <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}><T size={9}>{label}</T><View style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}><Icon name={ok ? 'checkmark-circle' : 'close-circle'} size={14} color={ok ? c.primary : c.danger} /><T size={8} color={ok ? c.primary : c.danger}>{ok ? 'Found' : 'Missing'}</T></View></View>;
}

function EmptyState({ icon, title, body, action, onPress }: { icon: string; title: string; body: string; action?: string; onPress?: () => void }) {
  const c = useContext(Theme);
  return <Card style={{ alignItems: 'center', paddingVertical: 28 }}><View style={{ width: 54, height: 54, borderRadius: 18, backgroundColor: c.tint, alignItems: 'center', justifyContent: 'center' }}><Icon name={icon} size={25} /></View><T size={16} bold style={{ textAlign: 'center' }}>{title}</T><T size={10} color={c.muted} style={{ textAlign: 'center' }}>{body}</T>{action && onPress ? <Button secondary compact title={action} onPress={onPress} /> : null}</Card>;
}
