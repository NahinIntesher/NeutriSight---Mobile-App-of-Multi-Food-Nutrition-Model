import { File } from 'expo-file-system';
import { fetch as expoFetch } from 'expo/fetch';
import Constants from 'expo-constants';
import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

export type Profile = {
  name: string;
  age: number | null;
  sex?: string | null;
  height_cm?: number | null;
  weight_kg?: number | null;
  allergies: string[];
  conditions: string[];
  preferences: string[];
  avoid: string[];
  goal: string;
  onboarding_complete?: boolean;
  onboarding_skipped?: boolean;
};

export type User = {
  id: string;
  name: string;
  email: string;
  profile: Partial<Profile>;
};

export type Food = {
  name: string;
  confidence: number;
  source: string;
  bbox: number[];
  polygon: number[][];
  yolo?: { name: string; confidence: number };
  suggestion?: { name: string; confidence: number } | null;
};

export type Result = {
  id: string;
  created_at: number;
  foods: Food[];
  nutrition: Record<string, number> | null;
  nutrition_scope?: string;
  warnings: string[];
  alerts: { level: string; title: string; reason: string }[];
  profile_required: boolean;
  saved: boolean;
  image_size: number[];
  models?: Record<string, unknown>;
};

let token = '';
let expired = () => {};

export const onExpired = (fn: () => void) => {
  expired = fn;
  return () => {
    expired = () => {};
  };
};

const inferredHost =
  Platform.OS === 'web' && typeof window !== 'undefined'
    ? window.location.hostname
    : Constants.expoConfig?.hostUri?.split(':')[0] || 'localhost';

export let baseURL = process.env.EXPO_PUBLIC_API_URL || `http://${inferredHost}:8000`;

export function setURL(value: string) {
  const parsed = new URL(value.trim());
  if (
    !['http:', 'https:'].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error('Enter a valid HTTP or HTTPS server address.');
  }
  baseURL = parsed.toString().replace(/\/$/, '');
}

export async function storeToken(value: string) {
  token = value;
  if (Platform.OS === 'web') {
    try {
      if (value) sessionStorage.setItem('nutrisight-token', value);
      else sessionStorage.removeItem('nutrisight-token');
    } catch {}
  } else if (value) {
    await SecureStore.setItemAsync('nutrisight-token', value);
  } else {
    await SecureStore.deleteItemAsync('nutrisight-token');
  }
}

export async function restoreToken() {
  if (Platform.OS === 'web') {
    try {
      token = sessionStorage.getItem('nutrisight-token') || '';
    } catch {
      token = '';
    }
  } else {
    token = (await SecureStore.getItemAsync('nutrisight-token')) || '';
  }
  return token;
}

function errorMessage(data: any, status: number) {
  if (typeof data?.detail === 'string') return data.detail;
  if (Array.isArray(data?.detail) && data.detail[0]?.msg) return data.detail[0].msg;
  if (status >= 500) return 'The server could not complete this request. Try again.';
  return 'Please check the entered information.';
}

async function request<T>(
  path: string,
  options: {
    method?: string;
    body?: any;
    headers?: Record<string, string>;
    timeout?: number;
    jsonBody?: boolean;
  } = {},
): Promise<T> {
  const auth = !['/api/auth/signup', '/api/auth/login', '/health'].includes(path);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeout ?? 20000);
  try {
    const response = await expoFetch(baseURL + path, {
      method: options.method || 'GET',
      headers: {
        ...(auth && token ? { Authorization: `Bearer ${token}` } : {}),
        ...(options.jsonBody ? { 'Content-Type': 'application/json' } : {}),
        ...(options.headers || {}),
      },
      body: options.body,
      signal: controller.signal,
    });

    const text = await response.text();
    let data: any = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { detail: text };
      }
    }

    if (!response.ok) {
      if (response.status === 401 && auth) {
        await storeToken('');
        expired();
      }
      throw new Error(errorMessage(data, response.status));
    }
    return data as T;
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new Error('Request timed out. Check the server and try again.');
    }
    if (
      ['Failed to fetch', 'Network request failed', 'Load failed'].includes(error?.message)
    ) {
      throw new Error('Cannot reach the server. Check the server address and Wi-Fi connection.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export async function api<T = any>(path: string, method = 'GET', body?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    body: body === undefined ? undefined : JSON.stringify(body),
    jsonBody: body !== undefined,
  });
}

export async function analyze(uri: string) {
  // Avoid multipart/FormData on native Expo. This fixes the
  // "Unsupported FormDataPart implementation" error seen on some runtimes.
  if (Platform.OS === 'web') {
    const blob = await (await globalThis.fetch(uri)).blob();
    return request<Result>('/api/analyze/raw', {
      method: 'POST',
      body: blob,
      headers: { 'Content-Type': blob.type || 'image/jpeg' },
      timeout: 180000,
    });
  }

  const file = new File(uri);
  return request<Result>('/api/analyze/raw', {
    method: 'POST',
    body: file,
    headers: {
      'Content-Type': file.type || 'image/jpeg',
      'X-File-Name': file.name || 'meal.jpg',
    },
    timeout: 180000,
  });
}
