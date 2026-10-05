import React, { useEffect, useState } from "react";
import { AccessibilityInfo, View } from "react-native";
import LottieView from "lottie-react-native";
import meal from "../assets/welcome.json";
import scanning from "../assets/scanning.json";
export default function Motion({
  size = 240,
  scan = false,
}: {
  size?: number;
  scan?: boolean;
}) {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then(setReduced);
    const s = AccessibilityInfo.addEventListener(
      "reduceMotionChanged",
      setReduced,
    );
    return () => s.remove();
  }, []);
  return (
    <View
      accessibilityLabel="Animated meal and nutrition illustration"
      style={{ alignItems: "center" }}
    >
      <LottieView
        source={scan ? scanning : meal}
        autoPlay={!reduced}
        loop={!reduced}
        progress={reduced ? 0.3 : undefined}
        style={{ width: size * 1.24, height: size }}
      />
    </View>
  );
}
