import React, { createContext, useContext, useState } from 'react';
import { Image, Pressable, Text, TextInput, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Circle, Ellipse, G, Line, Path, Rect } from 'react-native-svg';

export type Palette = {
  bg: string;
  card: string;
  elevated: string;
  text: string;
  muted: string;
  line: string;
  primary: string;
  primaryDeep: string;
  tint: string;
  lime: string;
  warn: string;
  danger: string;
  shadow: string;
  blue: string;
  orange: string;
};

export const light: Palette = {
  bg: '#FBFAF4',
  card: '#FFFFFF',
  elevated: '#FCFBF7',
  text: '#12372A',
  muted: '#738078',
  line: '#E9E8E1',
  primary: '#1B8A4B',
  primaryDeep: '#0F6B3A',
  tint: '#EEF6E9',
  lime: '#DDEEC4',
  warn: '#D68A18',
  danger: '#D65252',
  shadow: '#1E3F31',
  blue: '#3182CE',
  orange: '#EAA11A',
};

export const dark: Palette = {
  bg: '#0E1813',
  card: '#17231D',
  elevated: '#1D2B24',
  text: '#F4F6F2',
  muted: '#A9B4AD',
  line: '#304139',
  primary: '#7BC798',
  primaryDeep: '#63B383',
  tint: '#243A2D',
  lime: '#B8D694',
  warn: '#F0B45A',
  danger: '#F08A8A',
  shadow: '#000000',
  blue: '#73A9EA',
  orange: '#F2BB55',
};

export const Theme = createContext(light);

export function T({ children, size = 14, bold = false, color, style = {}, ...props }: any) {
  const c = useContext(Theme);
  return (
    <Text
      {...props}
      style={{
        fontFamily: bold ? 'Quicksand_700Bold' : 'Quicksand_500Medium',
        fontSize: Math.round(size * 1.2),
        lineHeight: Math.round(size * 1.2) * 1.3,
        color: color || c.text,
        ...style,
      }}
    >
      {children}
    </Text>
  );
}

export function Icon({ name, size = 22, color }: any) {
  const c = useContext(Theme);
  return <Ionicons name={name} size={size} color={color || c.primaryDeep} />;
}

export function Button({ title, onPress, secondary = false, icon, disabled = false, danger = false, compact = false }: any) {
  const c = useContext(Theme);
  const bg = danger ? '#FCE3E3' : secondary ? c.card : c.primaryDeep;
  const fg = danger ? '#BE3535' : secondary ? c.text : '#FFFFFF';
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={title}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => ({
        minHeight: compact ? 42 : 52,
        paddingHorizontal: compact ? 14 : 18,
        paddingVertical: compact ? 10 : 13,
        borderRadius: compact ? 12 : 13,
        backgroundColor: bg,
        opacity: disabled ? 0.48 : pressed ? 0.86 : 1,
        flexDirection: 'row',
        gap: 8,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: secondary ? 1 : 0,
        borderColor: secondary ? c.line : 'transparent',
        shadowColor: c.shadow,
        shadowOpacity: secondary ? 0.03 : 0.12,
        shadowRadius: 10,
        shadowOffset: { width: 0, height: 4 },
        elevation: secondary ? 1 : 2,
      })}
    >
      {icon ? <Icon name={icon} color={fg} size={compact ? 18 : 19} /> : null}
      <T bold size={compact ? 12 : 14} color={fg}>{title}</T>
    </Pressable>
  );
}

export function Card({ children, style = {} }: any) {
  const c = useContext(Theme);
  return (
    <View
      style={{
        backgroundColor: c.card,
        borderWidth: 1,
        borderColor: c.line,
        borderRadius: 16,
        padding: 16,
        gap: 12,
        shadowColor: c.shadow,
        shadowOpacity: 0.055,
        shadowRadius: 12,
        shadowOffset: { width: 0, height: 5 },
        elevation: 1,
        ...style,
      }}
    >
      {children}
    </View>
  );
}

export function Field({
  label,
  value,
  onChangeText,
  secure = false,
  keyboardType = 'default',
  placeholder = '',
  multiline = false,
  icon,
  autoComplete,
}: any) {
  const c = useContext(Theme);
  const [hidden, setHidden] = useState(secure);
  const [focused, setFocused] = useState(false);
  return (
    <View style={{ gap: 6 }}>
      <T size={12} bold>{label}</T>
      <View
        style={{
          minHeight: multiline ? 96 : 50,
          flexDirection: 'row',
          alignItems: multiline ? 'flex-start' : 'center',
          gap: 9,
          paddingHorizontal: 12,
          borderWidth: focused ? 1.4 : 1,
          borderColor: focused ? c.primary : c.line,
          borderRadius: 12,
          backgroundColor: c.elevated,
        }}
      >
        {icon ? <View style={{ paddingTop: multiline ? 13 : 0 }}><Icon name={icon} size={17} color={focused ? c.primary : c.muted} /></View> : null}
        <TextInput
          accessibilityLabel={label}
          value={value}
          onChangeText={onChangeText}
          secureTextEntry={hidden}
          keyboardType={keyboardType}
          autoCapitalize={secure || keyboardType === 'email-address' ? 'none' : 'sentences'}
          autoCorrect={!secure && keyboardType !== 'email-address'}
          autoComplete={autoComplete}
          placeholder={placeholder}
          placeholderTextColor={c.muted}
          multiline={multiline}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          style={{
            flex: 1,
            fontFamily: 'Quicksand_500Medium',
            fontSize: 14,
            color: c.text,
            paddingVertical: 11,
            minHeight: multiline ? 90 : 46,
            textAlignVertical: multiline ? 'top' : 'center',
          }}
        />
        {secure ? (
          <Pressable onPress={() => setHidden(!hidden)} hitSlop={10}>
            <Icon name={hidden ? 'eye-outline' : 'eye-off-outline'} size={18} color={c.muted} />
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

export function LogoMark({ size = 54 }: { size?: number }) {
  return <Image source={require('../assets/mock/logo.png')} resizeMode="contain" style={{ width: size, height: size }} />;
}

export function Brand({ large = false, centered = false, tagline = false }: { large?: boolean; centered?: boolean; tagline?: boolean }) {
  const c = useContext(Theme);
  return (
    <View style={{ alignItems: centered ? 'center' : 'flex-start', gap: large ? 4 : 1 }}>
      <View style={{ flexDirection: centered ? 'column' : 'row', alignItems: 'center', gap: centered ? 2 : 7 }}>
        <LogoMark size={large ? 78 : 34} />
        <View style={{ flexDirection: 'row', alignItems: 'baseline' }}>
          <T size={large ? 30 : 20} bold color="#0B5D3B">Nutri</T>
          <T size={large ? 30 : 20} bold color="#2E8B3E">Sight</T>
        </View>
      </View>
      {tagline ? <T size={10} color={c.muted} style={{ textAlign: centered ? 'center' : 'left' }}>See Your Food. Know Your Nutrition.</T> : null}
    </View>
  );
}

export function ServerBadge({ status, label, onPress }: { status: 'checking' | 'online' | 'offline'; label?: string; onPress?: () => void }) {
  const c = useContext(Theme);
  const color = status === 'online' ? c.primary : status === 'offline' ? c.danger : c.warn;
  const text = label || (status === 'online' ? 'Connected' : status === 'offline' ? 'Offline' : 'Checking');
  return (
    <Pressable onPress={onPress} disabled={!onPress} style={({ pressed }) => ({ opacity: pressed ? 0.75 : 1 })}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: c.card, borderColor: c.line, borderWidth: 1, paddingHorizontal: 9, paddingVertical: 6, borderRadius: 99 }}>
        <View style={{ width: 7, height: 7, borderRadius: 7, backgroundColor: color }} />
        <T size={9} bold color={color}>{text}</T>
      </View>
    </Pressable>
  );
}

export function FoodPhoneArt({ size = 250 }: { size?: number }) {
  return <Image source={require('../assets/mock/intro_art.png')} resizeMode="contain" style={{ width: size, height: size * 0.97 }} />;
}

export function SplashMealArt({ size = 300 }: { size?: number }) {
  return <Image source={require('../assets/mock/splash_bowl.png')} resizeMode="contain" style={{ width: size, height: size * 0.69, borderRadius: 24 }} />;
}

export function AnalyzeArt({ size = 210 }: { size?: number }) {
  return <Image source={require('../assets/mock/analyze_art.png')} resizeMode="contain" style={{ width: size, height: size * 0.73 }} />;
}

function TelescopeLabel() {
  return (
    <G>
      <Circle cx="164" cy="105" r="22" fill="#E4F0D8" />
      <Path d="M151 110c9-18 22-20 31-8-6 15-17 21-31 16Z" fill="#6B9E48" />
    </G>
  );
}

export function AvatarArt({ size = 68 }: { size?: number }) {
  return <Image source={require('../assets/mock/avatar.png')} style={{ width: size, height: size, borderRadius: size / 2 }} />;
}

export function MacroDonut({ protein, carbs, fat, size = 124 }: { protein: number; carbs: number; fat: number; size?: number }) {
  const c = useContext(Theme);
  const pKcal = Math.max(0, protein) * 4;
  const cKcal = Math.max(0, carbs) * 4;
  const fKcal = Math.max(0, fat) * 9;
  const total = pKcal + cKcal + fKcal || 1;
  const p = pKcal / total;
  const carb = cKcal / total;
  const r = 44;
  const circ = 2 * Math.PI * r;
  return (
    <Svg width={size} height={size} viewBox="0 0 120 120">
      <G rotation="-90" origin="60,60">
        <Circle cx="60" cy="60" r={r} stroke={c.line} strokeWidth="18" fill="none" />
        <Circle cx="60" cy="60" r={r} stroke={c.primaryDeep} strokeWidth="18" fill="none" strokeDasharray={`${circ * p} ${circ * (1 - p)}`} strokeLinecap="butt" />
        <Circle cx="60" cy="60" r={r} stroke={c.blue} strokeWidth="18" fill="none" strokeDasharray={`${circ * carb} ${circ * (1 - carb)}`} strokeDashoffset={-circ * p} strokeLinecap="butt" />
        <Circle cx="60" cy="60" r={r} stroke={c.orange} strokeWidth="18" fill="none" strokeDasharray={`${circ * Math.max(0, 1 - p - carb)} ${circ * (p + carb)}`} strokeDashoffset={-circ * (p + carb)} strokeLinecap="butt" />
      </G>
      <Circle cx="60" cy="60" r="28" fill={c.card} />
    </Svg>
  );
}
