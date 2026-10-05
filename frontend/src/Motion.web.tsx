import React from 'react';
import { View } from 'react-native';
import Svg, { Circle, Path, Rect } from 'react-native-svg';

export default function Motion({ size = 240, scan = false }: { size?: number; scan?: boolean }) {
  const w = size * 1.24;
  const h = size;
  return (
    <View accessibilityLabel={scan ? 'Meal scanning illustration' : 'Meal nutrition illustration'} style={{ width: w, height: h, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={w} height={h} viewBox="0 0 360 290">
        <Circle cx="180" cy="150" r="112" fill="#EAF2E6" />
        <Circle cx="180" cy="154" r="84" fill="#FFFFFF" stroke="#DCE7DB" strokeWidth="4" />
        <Path d="M116 137c16-34 58-46 89-27 32 20 38 63 13 88-28 28-76 22-94-12-9-17-11-33-8-49Z" fill="#E8B670" />
        <Path d="M199 103c29 4 49 28 48 58-18 4-35-2-47-17-10-12-11-27-1-41Z" fill="#6D9E60" />
        <Circle cx="126" cy="181" r="21" fill="#8DBA78" />
        <Circle cx="151" cy="198" r="18" fill="#5F9155" />
        <Path d="M207 191c17-16 39-13 51 5-12 21-39 25-57 9 0-5 2-10 6-14Z" fill="#D77C5D" />
        {scan ? <>
          <Path d="M76 83h28M76 83v28M284 83h-28M284 83v28M76 225h28M76 225v-28M284 225h-28M284 225v-28" stroke="#1D4F39" strokeWidth="7" strokeLinecap="round" />
          <Rect x="96" y="147" width="168" height="4" rx="2" fill="#2B684B" opacity="0.85" />
        </> : <>
          <Rect x="230" y="52" width="78" height="72" rx="22" fill="#FFFFFF" stroke="#DCE7DB" strokeWidth="3" />
          <Path d="M252 96V78M270 96V68M288 96V60" stroke="#2B684B" strokeWidth="7" strokeLinecap="round" />
        </>}
      </Svg>
    </View>
  );
}
