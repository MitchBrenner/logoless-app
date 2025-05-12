// Copyright (c) 2025 Mitchell Brenner
// Licensed under the GNU General Public License v3.0 (GPL-3.0-or-later)
// See LICENSE for details.

import { Stack } from "expo-router";

export default function RootLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: false,
      }}
    />
  );
}
